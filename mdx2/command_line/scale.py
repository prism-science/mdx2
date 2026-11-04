"""
Fit scaling model to unmerged corrected intensities
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from loguru import logger
from simple_parsing import field

from mdx2 import profiling
from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.data import HKLTable
from mdx2.io import loadobj, saveobj
from mdx2.scaling import BatchModelRefiner, ScaledData


@dataclass
class PrescaleParameters:
    """Options for initial quick refinement of models to a subset of data and/or isotropically averaged data"""

    enable: bool = False  # include pre-scaling refinement
    subsample: int = 10  # subsample factor for pre-scaling refinement
    isotropic: bool = True  # use isotropically averaged data for pre-scaling refinement
    scaling: bool = True  # include scaling model in pre-scaling refinement if scaling.enable is True
    offset: bool = False  # include offset model in pre-scaling refinement if offset.enable is True
    detector: bool = False  # include detector model in pre-scaling refinement if detector.enable is True
    absorption: bool = True  # include absorption model in pre-scaling refinement if absorption.enable is True


@dataclass
class ScalingModelParameters:
    """Options to control refinement of the scaling model"""

    enable: bool = True  # include smooth scale factor vs. phi
    alpha: float = 1.0  # amount to rescale the default smoothness (regularization parameter)
    dphi: float = 1.0  # spacing of phi control points in degrees
    niter: int = 10  # maximum iterations in refinement
    x2tol: float = 1.0e-4  # maximum change in x2 to stop refinement early
    outlier: float = 10.0  # standard error cutoff for outlier rejection after refinement


@dataclass
class OffsetModelParameters:
    """Options to control refinement of the offset model"""

    enable: bool = False  # include smooth offset vs. resolution and phi
    alpha_x: float = 1.0  # smoothness vs. s (resolution), multiplies the regularization parameter
    alpha_y: float = 1.0  # smoothness vs. phi, multiplies the regularization parameter
    alpha_min: float = 0.001  # deviation from offset.min, multiplies the regularization parameter
    min: float = 0.0  # minimum value of offset
    dphi: float = 2.5  # spacing of phi control points in degrees
    ns: int = 31  # number of s (resolution) control points
    niter: int = 5  # maximum iterations in refinement
    x2tol: float = 1.0e-3  # maximum change in x2 to stop refinement early
    outlier: float = 5.0  # standard error cutoff for outlier rejection after refinement


@dataclass
class DetectorModelParameters:
    """Options to control refinement of the detector model"""

    enable: bool = False  # include smooth scale vs. detector xy position
    alpha: float = 1.0  # smoothness vs. xy position: multiplies the regularization parameter
    nx: int = 200  # number of grid control points in the x direction
    ny: int = 200  # number of grid control points in the y direction
    niter: int = 5  # maximum iterations in refinement
    x2tol: float = 1.0e-3  # maximum change in x2 to stop refinement early
    outlier: float = 5.0  # standard error cutoff for outlier rejection after refinement


@dataclass
class AbsorptionModelParameters:
    """Options to control refinement of the absorption model"""

    enable: bool = False  # include smooth scale vs. detector xy position and phi
    alpha_xy: float = 10.0  # smoothness vs. xy position: multiplies the regularization parameter
    alpha_z: float = 1.0  # smoothness vs. phi: multiplies the regularization parameter
    nx: int = 20  # number of grid control points in the x direction
    ny: int = 20  # number of grid control points in the y direction
    dphi: float = 5.0  # spacing of phi control points in degrees
    niter: int = 5  # maximum iterations in refinement
    x2tol: float = 1.0e-4  # maximum change in x2 to stop refinement early
    outlier: float = 5.0  # standard error cutoff for outlier rejection after refinement


@dataclass
class Parameters:
    """Options for refining a scaling model to unmerged corrected intensities"""

    hkl: List[str] = field(positional=True, nargs="+")  # NeXus file(s) containing hkl_table
    prescale: PrescaleParameters = field(default_factory=PrescaleParameters)
    scaling: ScalingModelParameters = field(default_factory=ScalingModelParameters)
    absorption: AbsorptionModelParameters = field(default_factory=AbsorptionModelParameters)
    detector: DetectorModelParameters = field(default_factory=DetectorModelParameters)
    offset: OffsetModelParameters = field(default_factory=OffsetModelParameters)
    outfile: Optional[List[str]] = field(default=None, nargs="*")
    """name of the output NeXus file(s). If omitted, will attempt a sensible name such as scales.nxs"""
    logfile: str = "mdx2.scale.log"  # name of the log file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)
    engine: str = "default"
    """scaling engine: "default" (mdx2.scaling) or "fast" (same algorithm, compiled multi-threaded passes over the
    data; needs numba; threads from --nproc). Prescaling is not supported by the fast engine."""
    profile: bool = False  # log a summary of wall time and memory per step (load, apply, merge, fit, ...)
    profile_file: Optional[str] = None  # write wall time and memory per step to this file (JSON lines)
    mca2020: bool = False
    """shortcut for --scaling.enable True --offset.enable True --detector.enable True --absorption.enable True"""

    def __post_init__(self):
        """Validate and process parameters after initialization"""
        # Apply mca2020 shortcut to enable all models
        if self.mca2020:
            self.scaling.enable = True
            self.detector.enable = True
            self.absorption.enable = True
            self.offset.enable = True

        # Check for duplicate input files
        # Note: Same filename in different directories is valid (e.g., /path1/data.nxs, /path2/data.nxs)
        # But truly duplicate paths are not allowed (e.g., data.nxs, data.nxs)
        if self.engine not in ("default", "fast"):
            raise ValueError(f"Unknown engine {self.engine!r} (use 'default' or 'fast')")
        if self.engine == "fast" and self.prescale.enable:
            raise ValueError("--prescale.enable is not supported with --engine fast")
        if len(self.hkl) != len(set(self.hkl)):
            raise ValueError("Duplicate input files are not allowed")

        # Auto-generate output file names if not provided
        # This handles various patterns:
        # - Different directories → scales.nxs in each directory
        # - Same directory with underscore pattern → scales_<postfix>.nxs
        # - Other cases return None and trigger error below
        if self.outfile is None:
            self.outfile = generate_default_outfiles(self.hkl)
            if self.outfile is None:
                raise ValueError("unable to auto-generate output file names from input name pattern")

        # Validate that the number of output files matches the number of input files
        if len(self.outfile) != len(self.hkl):
            raise ValueError(
                f"Number of output files ({len(self.outfile)}) must match number of input files ({len(self.hkl)})"
            )

        # Ensure at least one model is enabled
        if not (self.scaling.enable or self.absorption.enable or self.detector.enable or self.offset.enable):
            raise ValueError("At least one model must be enabled: scaling, absorption, detector, or offset.")


def generate_default_outfiles(infiles):
    """Generate default output file names based on input file names.

    - If the input files are in different directories, returns a list of scales.nxs in each directory.
    - If the input files are in the same directory and all have an underscore in the root name,
      and the postfixes (the part after the last underscore) are unique, returns a list of
      scales_<postfix>.nxs in the same directory.
    - Otherwise, returns None.
    """
    dirs = [os.path.dirname(fn) for fn in infiles]
    if len(set(dirs)) == len(dirs):  # dirs are unique
        return [os.path.join(d, "scales.nxs") for d in dirs]
    if len(set(dirs)) == 1:  # dirs are identical
        roots = [os.path.splitext(os.path.split(fn)[-1])[0] for fn in infiles]
        if all("_" in root for root in roots):
            postfix = [root.split("_")[-1] for root in roots]
            if len(set(postfix)) == len(postfix):  # postfixes are unique
                return [os.path.join(dirs[0], f"scales_{pf}.nxs") for pf in postfix]
    return None


def mask_outliers(MR, outlier):
    """Mask outliers"""
    logger.info("    Applying scale factors...")
    MR.apply()
    logger.info("    Merging...")
    Im, _sigmam, _counts = MR.data.merge()
    nout = MR.data.mask_outliers(Im, outlier)
    logger.info("    Removed {} outliers (>{} sigma)", nout, outlier)


def refine_offset_model(MR, offset_params):
    """Refine the offset model"""
    old_x2 = 1e6  # initialize to some large number
    for j in range(offset_params.niter):
        logger.info("  Iteration {}/{}", j + 1, offset_params.niter)
        logger.info("    Applying scale factors...")
        MR.apply()
        logger.info("    Merging...")
        Im, _sigmam, _counts = MR.data.merge()

        logger.info("    Fitting the model...")
        x2 = MR.cfit(
            Im,
            offset_params.alpha_x,
            offset_params.alpha_y,
            offset_params.alpha_min,
            offset_params.min,
        )  # 1,1,.1,min_c=0
        logger.info("    χ²: {:.6f}", x2)
        if old_x2 - x2 < offset_params.x2tol:
            logger.info("    Converged (Δχ² < {})", offset_params.x2tol)
            break
        old_x2 = x2


def refine_scaling_model(MR, scaling_params):
    """Refine the scaling model"""
    old_x2 = 1e6  # initialize to some large number
    for j in range(scaling_params.niter):
        logger.info("  Iteration {}/{}", j + 1, scaling_params.niter)
        logger.info("    Applying scale factors...")
        MR.apply()
        logger.info("    Merging...")
        Im, _sigmam, _counts = MR.data.merge()
        logger.info("    Fitting the model...")
        x2 = MR.bfit(Im, scaling_params.alpha)
        logger.info("    χ²: {:.6f}", x2)
        if old_x2 - x2 < scaling_params.x2tol:
            logger.info("    Converged (Δχ² < {})", scaling_params.x2tol)
            break
        old_x2 = x2


def refine_scaling_and_offset_model(MR, scaling_params, offset_params):
    """Refine the scaling and offset model"""
    old_x2 = 1e6  # initialize to some large number
    for j in range(offset_params.niter):
        logger.info("  Iteration {}/{}", j + 1, offset_params.niter)
        logger.info("    Applying scale factors...")
        MR.apply()
        logger.info("    Merging...")
        Im, _sigmam, _counts = MR.data.merge()
        logger.info("    Fitting the model...")
        MR.cfit(
            Im,
            offset_params.alpha_x,
            offset_params.alpha_y,
            offset_params.alpha_min,
            offset_params.min,
        )  # 1,1,.1,min_c=0
        x2 = MR.bfit(Im, scaling_params.alpha)
        logger.info("    χ²: {:.6f}", x2)
        if old_x2 - x2 < offset_params.x2tol:
            logger.info("    Converged (Δχ² < {})", offset_params.x2tol)
            break
        old_x2 = x2


def refine_detector_model(MR, detector_params):
    """Refine the detector model"""
    old_x2 = 1e6  # initialize to some large number
    for j in range(detector_params.niter):
        logger.info("  Iteration {}/{}", j + 1, detector_params.niter)
        logger.info("    Applying scale factors...")
        MR.apply()
        logger.info("    Merging...")
        Im, _sigmam, _counts = MR.data.merge()
        logger.info("    Fitting the model...")
        x2 = MR.dfit(Im, detector_params.alpha)
        logger.info("    χ²: {:.6f}", x2)
        if old_x2 - x2 < detector_params.x2tol:
            logger.info("    Converged (Δχ² < {})", detector_params.x2tol)
            break
        old_x2 = x2


def refine_absorption_model(MR, absorption_params):
    """Refine the absorption model"""
    old_x2 = 1e6  # initialize to some large number
    for j in range(absorption_params.niter):
        logger.info("  Iteration {}/{}", j + 1, absorption_params.niter)
        logger.info("    Applying scale factors...")
        MR.apply()
        logger.info("    Merging...")
        Im, _sigmam, _counts = MR.data.merge()
        logger.info("    Fitting the model...")
        x2 = MR.afit(Im, absorption_params.alpha_xy, absorption_params.alpha_z)
        logger.info("    χ²: {:.6f}", x2)
        if old_x2 - x2 < absorption_params.x2tol:
            logger.info("    Converged (Δχ² < {})", absorption_params.x2tol)
            break
        old_x2 = x2


def load_data_for_scaling(*hkl_files, subsample=None, merge_isotropic=False, engine="default"):
    """Load data from hkl files into a single HKLTable"""
    P = profiling.PROFILER
    tabs = []
    for n, fn in enumerate(hkl_files):
        with P.step("load", file=fn):
            tmp = loadobj(fn, "hkl_table")
        if subsample is not None:
            tmp = tmp[::subsample]
        tmp.batch = n * np.ones_like(tmp.op)
        tabs.append(tmp)

    with P.step("concatenate"):
        hkl = HKLTable.concatenate(tabs)

    logger.info("Loaded {} reflections from {} file(s)", len(hkl), len(hkl_files))

    if merge_isotropic:
        # TODO: should bin width be a parameter?
        # (here we use 0.001 as a default bin width)
        logger.info("Grouping by scattering vector length (isotropic averaging)...")
        index_map = np.floor(hkl.s * 1000).astype(np.uint16)
    else:
        logger.info("Grouping redundant observations...")
        with P.step("group"):
            index_map = _group_index(hkl)

    if engine == "fast":
        from mdx2.fast_scaling import FastScaledData as data_class  # noqa: PLC0415 (numba is optional)
    else:
        data_class = ScaledData
    with P.step("init_data"):
        S = data_class(
            hkl.intensity,
            hkl.intensity_error,
            index_map,
            phi=hkl.phi,
            s=hkl.s,
            ix=hkl.ix,
            iy=hkl.iy,
            batch=hkl.batch,
        )

    with P.step("mask_singletons"):
        nsingletons = S.mask_singletons()
    if nsingletons > 0:
        logger.info("Masked {} singletons (reflections with only one observation)", nsingletons)

    return S


def run_scale(params):
    """Run the scale algorithm"""

    profiling.PROFILER = P = profiling.Profiler(params.profile, params.profile_file)

    S = load_data_for_scaling(*params.hkl, engine=params.engine)

    if params.engine == "fast":
        # threads start after loading and grouping (plain numpy), just before the first kernel
        _set_threads(params.nproc)
        from mdx2.fast_scaling import FastBatchModelRefiner  # noqa: PLC0415 (numba is optional)

        MR = FastBatchModelRefiner(S)
    else:
        MR = BatchModelRefiner(S)
        if params.nproc != 1:
            logger.warning("Serial execution only, ignoring nproc value")
    _instrument(MR, P)

    if params.scaling.enable:
        MR.add_scaling_models(
            dphi=params.scaling.dphi,
        )
    if params.offset.enable:
        MR.add_offset_models(
            dphi=params.offset.dphi,
            ns=params.offset.ns,
        )
    if params.absorption.enable:
        MR.add_absorption_models(
            dphi=params.absorption.dphi,
            nix=params.absorption.nx,
            niy=params.absorption.ny,
        )
    if params.detector.enable:
        MR.add_detector_model(
            nix=params.detector.nx,
            niy=params.detector.ny,
        )

    # prescaling phase
    if params.prescale.enable:
        if params.prescale.subsample > 1:
            subsample = params.prescale.subsample
        else:
            subsample = None
        S_pre = load_data_for_scaling(*params.hkl, subsample=subsample, merge_isotropic=params.prescale.isotropic)
        MR_pre = BatchModelRefiner(S_pre)
        _instrument(MR_pre, P)
        # transfer models to prescaling refiner
        for model_refiner, model_refiner_pre in zip(MR.batch_refiners, MR_pre.batch_refiners):
            model_refiner_pre.scaling.model = model_refiner.scaling.model
            model_refiner_pre.offset.model = model_refiner.offset.model
            model_refiner_pre.detector.model = model_refiner.detector.model
            model_refiner_pre.absorption.model = model_refiner.absorption.model
        # refine models on prescaling data
        logger.info("Starting prescaling refinement...")
        if params.prescale.scaling and params.scaling.enable:
            logger.info("Optimizing scale vs. phi (b)...")
            P.stage = "pre b"
            refine_scaling_model(MR_pre, params.scaling)
            mask_outliers(MR_pre, params.scaling.outlier)
        if params.prescale.offset and params.prescale.scaling and params.offset.enable and params.scaling.enable:
            logger.info("Optimizing scaling and offset models together (b,c)...")
            P.stage = "pre b,c"
            refine_scaling_and_offset_model(MR_pre, params.scaling, params.offset)
            mask_outliers(MR_pre, params.offset.outlier)
        if params.prescale.offset and params.offset.enable:
            logger.info("Optimizing offset vs. phi and resolution (c)...")
            P.stage = "pre c"
            refine_offset_model(MR_pre, params.offset)
            mask_outliers(MR_pre, params.offset.outlier)
        if params.prescale.detector and params.detector.enable:
            logger.info("Optimizing scale vs. detector position (d)...")
            P.stage = "pre d"
            refine_detector_model(MR_pre, params.detector)
            mask_outliers(MR_pre, params.detector.outlier)
        if params.prescale.absorption and params.absorption.enable:
            logger.info("Optimizing scale vs. detector position and phi (a)...")
            P.stage = "pre a"
            refine_absorption_model(MR_pre, params.absorption)
            mask_outliers(MR_pre, params.absorption.outlier)
        logger.info("Prescaling refinement completed")

    logger.info("Starting main scaling refinement...")

    if params.scaling.enable:
        logger.info("Optimizing scale vs. phi (b)...")
        P.stage = "b"
        refine_scaling_model(MR, params.scaling)
        mask_outliers(MR, params.scaling.outlier)

    if params.scaling.enable and params.offset.enable:
        logger.info("Optimizing scaling and offset models together (b,c)...")
        P.stage = "b,c"
        refine_scaling_and_offset_model(MR, params.scaling, params.offset)
        mask_outliers(MR, params.offset.outlier)

    if params.offset.enable:
        logger.info("Optimizing offset vs. phi and resolution (c)...")
        P.stage = "c"
        refine_offset_model(MR, params.offset)
        mask_outliers(MR, params.offset.outlier)

    if params.detector.enable:
        logger.info("Optimizing scale vs. detector position (d)...")
        P.stage = "d"
        refine_detector_model(MR, params.detector)
        mask_outliers(MR, params.detector.outlier)

    if params.absorption.enable:
        logger.info("Optimizing scale vs. detector position and phi (a)...")
        P.stage = "a"
        refine_absorption_model(MR, params.absorption)
        mask_outliers(MR, params.absorption.outlier)

    logger.info("Refinement completed")

    logger.info("Saving scaling models...")
    P.stage = None
    with P.step("save"):
        _save_models(MR, params.outfile)
    P.summary()

    logger.info("Scaling completed successfully")


def _group_index(hkl, max_table_ratio=4):
    """Group index of each observation: the same as HKLTable.unique()'s index_map (groups numbered in order of the
    packed (H, K, L) key), computed in linear time with a table over the key range when that range is at most
    max_table_ratio x the number of observations (table memory ~9 bytes per possible key); otherwise np.unique."""
    P = profiling.PROFILER
    with P.step("to_array_index"):
        ind, _ = hkl.to_array_index()
    with P.step("pack_keys"):
        sz = [int(np.max(j)) + 1 for j in ind]
        key = np.ravel_multi_index(ind, sz)
    n_keys = int(np.prod(sz, dtype=np.int64))
    if n_keys <= max_table_ratio * key.size:
        with P.step("key_table", n_keys=n_keys):
            present = np.zeros(n_keys, dtype=bool)
            present[key] = True
            group_of_key = np.cumsum(present, dtype=np.int64) - 1
            index_map = group_of_key[key]
        logger.info("Grouping with a key table ({} possible keys, {} groups)", n_keys, int(group_of_key[-1]) + 1)
        return index_map
    logger.info("Grouping by sorting ({} possible keys > {} x {} observations)", n_keys, max_table_ratio, key.size)
    with P.step("sort_unique", n_keys=n_keys):
        return np.unique(key, return_inverse=True)[1].ravel()


def _set_threads(nproc):
    """numba threads for the fast engine, with mdx2's nproc convention (1 = sequential, -1 = all, -N = all but N-1).

    Uses numba's workqueue threading layer unless NUMBA_THREADING_LAYER is set: numba 0.65.1's GNU OpenMP layer was
    seen executing parallel-loop iterations more than once on a 96-core Linux node (wrong results, very slow)."""
    import numba  # noqa: PLC0415 (optional dependency, fast engine only)

    if "NUMBA_THREADING_LAYER" not in os.environ:
        numba.config.THREADING_LAYER = "workqueue"
    ncpu = numba.config.NUMBA_NUM_THREADS
    n = nproc if nproc > 0 else max(1, ncpu + 1 + nproc)
    numba.set_num_threads(min(n, ncpu))
    logger.info(
        "Fast engine: {} thread(s), numba threading layer {}", numba.get_num_threads(), numba.config.THREADING_LAYER
    )


def _instrument(MR, P):
    """Time the refiner's operations (no-op unless profiling is enabled)."""
    for m, model in (
        ("add_scaling_models", "b"),
        ("add_offset_models", "c"),
        ("add_absorption_models", "a"),
        ("add_detector_model", "d"),
    ):
        P.wrap(MR, m, name="add_models", model=model)
    for m in ("apply", "bfit", "afit", "cfit", "dfit"):
        P.wrap(MR, m, name="fit_" + m[0] if m.endswith("fit") else m)
    P.wrap(MR.data, "merge")
    P.wrap(MR.data, "mask_outliers")


def _save_models(MR, outfiles):
    for model_refiner, fn in zip(MR.batch_refiners, outfiles):
        models = dict(
            scaling_model=model_refiner.scaling.model,
            detector_model=model_refiner.detector.model,
            absorption_model=model_refiner.absorption.model,
            offset_model=model_refiner.offset.model,
        )

        file_created = False
        for name, model in models.items():
            if model is not None:
                saveobj(model, fn, name=name, append=file_created)
                file_created = True


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_scale))


if __name__ == "__main__":
    run()
