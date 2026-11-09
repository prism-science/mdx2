"""
Merge corrected intensities using a scaling model
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from loguru import logger
from nexusformat.nexus import nxload
from simple_parsing import field  # pip install simple-parsing

from mdx2 import profiling
from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.data import HKLTable
from mdx2.io import loadobj, saveobj
from mdx2.parallel import set_threads
from mdx2.scaling import BatchModelRefiner, ScaledData


@dataclass
class Parameters:
    """Options for merging scaled intensities"""

    hkl: List[str] = field(positional=True, nargs="+")  # NeXus file(s) containing hkl_table
    scale: List[str] = field(nargs="+")  # NeXus file(s) with scaling models
    outlier: Optional[float] = None  # optional standard error cutoff for outlier rejection
    outlier_method: str = "two_sided"
    """"two_sided" (default; per symmetry-equivalent group, the observation with the largest leave-one-out residual,
    repeated until none is beyond the cutoff; both observations of a disagreeing pair) or "legacy" (mdx2 <= 1.0.5:
    observations below the merged value only, one pass)"""
    split: Optional[str] = field(default=None, choices=["randomHalf", "weightedRandomHalf", "Friedel"])
    """also merge data into separate columns based on splitting criteria"""
    geometry: Optional[str] = (
        None  # NeXus file containing the Laue group symmetry operators, required for --split Friedel
    )
    outfile: str = "merged.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.merge.log"  # name of the log file
    scaling: bool = field(default=True, negative_prefix="--no-")  # apply scaling model if present
    offset: bool = field(default=True, negative_prefix="--no-")  # apply offset model if present
    absorption: bool = field(default=True, negative_prefix="--no-")  # apply absorption model if present
    detector: bool = field(default=True, negative_prefix="--no-")  # apply detector model if present
    nproc: int = -1
    """threads: -1 = all available CPUs (respecting the affinity mask and container CPU quota), N = N threads,
    -N = all available CPUs but N-1"""
    profile: bool = False  # log a summary of wall time and memory per step (load, group, apply, merge, ...)
    profile_file: Optional[str] = None  # write wall time and memory per step to this file (JSON lines)

    def __post_init__(self):
        """Validate inter-parameter dependencies and outlier parameter"""
        if self.split == "Friedel" and self.geometry is None:
            raise ValueError("--geometry argument is required for symmetry-based splitting")
        if self.outlier_method not in ("two_sided", "legacy"):
            raise ValueError(f"Unknown outlier_method {self.outlier_method!r} (use 'two_sided' or 'legacy')")
        if self.outlier is not None and self.outlier <= 0:
            raise ValueError(f"outlier must be > 0, got {self.outlier}")
        if len(self.scale) != len(self.hkl):
            raise ValueError(
                f"Number of scale files ({len(self.scale)}) must match number of HKL inputs ({len(self.hkl)})"
            )


MAX_OUTLIER_ROUNDS = 100  # per-group cap for two-sided rejection (insulin: 7 rounds at 5 sigma, 13 at 3 sigma)

# note: negative_prefix is used to allow --no-scaling, --no-offset, etc. for consistency with old argparse api
# future: could switch split to an Enum type, to simplify selection of choices by alternate APIs


def run_merge(params):
    """Run the merge script"""
    profiling.PROFILER = P = profiling.Profiler(params.profile, params.profile_file)

    # load data into a giant table
    logger.info("Loading {} HKL table(s)...", len(params.hkl))
    tabs = []
    for n, fn in enumerate(params.hkl):
        with P.step("load", file=fn):
            tmp = loadobj(fn, "hkl_table")
        tmp.batch = n * np.ones_like(tmp.op)
        tabs.append(tmp)
    with P.step("concatenate"):
        hkl = HKLTable.concatenate(tabs)
    logger.info("Total reflections loaded: {}", len(hkl))

    logger.info("Grouping redundant observations...")
    with P.step("group"):
        (h, k, l), index_map, counts = hkl.unique()

    with P.step("init_data"):
        S = ScaledData(
            hkl.intensity,
            hkl.intensity_error,
            index_map,
            phi=hkl.phi,
            s=hkl.s,
            ix=hkl.ix,
            iy=hkl.iy,
            batch=hkl.batch,
        )
    set_threads(params.nproc)
    MR = BatchModelRefiner(S)

    logger.info("Loading scaling models...")
    use = dict(absorption=params.absorption, offset=params.offset, detector=params.detector, scaling=params.scaling)
    with P.step("load_models"):
        for batch, fn in enumerate(params.scale):
            entry = nxload(fn).entry.keys()
            for name, key in (("scaling", "b"), ("absorption", "a"), ("offset", "c"), ("detector", "d")):
                if use[name] and f"{name}_model" in entry:
                    MR.set_model(batch, key, loadobj(fn, f"{name}_model"))

    logger.info("Applying scale factors...")
    with P.step("apply"):
        MR.apply()
    logger.info("Merging observations...")
    with P.step("merge"):
        Im, sigmam, counts = S.merge()

    if params.outlier is not None:
        with P.step("mask_outliers", method=params.outlier_method):
            if params.outlier_method == "legacy":
                nout = S.mask_outliers(Im, params.outlier, method="legacy")
                logger.info("Removed {} outliers (>{} sigma)", nout, params.outlier)
            else:
                # rounds repeat within each group (one pass over the data), up to MAX_OUTLIER_ROUNDS per group
                nout = S.mask_outliers(None, params.outlier, max_rounds=MAX_OUTLIER_ROUNDS)
                o = S.last_outliers
                if len(o["capped"]):
                    worst = ", ".join(f"({h[g]:g}, {k[g]:g}, {l[g]:g})" for g in o["capped"][:5])
                    logger.warning(
                        "Outlier rejection stopped after {} rounds in {} group(s), still finding outliers: {}{}",
                        MAX_OUTLIER_ROUNDS,
                        len(o["capped"]),
                        worst,
                        ", ..." if len(o["capped"]) > 5 else "",
                    )
                logger.info(
                    "Removed {} outliers (|leave-one-out residual| >{} sigma; {} high, {} low; {} round(s))",
                    nout,
                    params.outlier,
                    o["high"],
                    o["low"],
                    o["rounds"],
                )
        logger.info("Merging again...")
        with P.step("merge"):
            Im, sigmam, counts = S.merge()

    cols = dict(
        intensity=Im.astype(np.float32),
        intensity_error=sigmam.astype(np.float32),
        count=counts.astype(np.int32),
    )

    split = params.split
    if split is not None:
        with P.step("split", mode=split):
            if split in ("weightedRandomHalf", "randomHalf"):
                if split == "weightedRandomHalf":
                    logger.info("Splitting by weighted random half algorithm...")
                    w = S.weights()
                else:
                    logger.info("Splitting into random half-datasets (unweighted)...")
                    w = np.ones(len(S))
                # same random draws as mdx2 <= 1.0.5 (global numpy random state)
                jitter = np.random.random_sample(len(S)) * 0.2
                flips = np.random.randint(2, size=len(h))
                isgrp1 = S.split_half(w, jitter, flips)
                groups = [isgrp1, ~isgrp1]
            elif split == "Friedel":
                logger.info("Splitting into Friedel pairs...")
                symm = loadobj(params.geometry, "symmetry")
                has_inversion = np.array([np.linalg.det(op) for op in symm.laue_group_operators]) < 0
                isminus = has_inversion[hkl.op]
                groups = [~isminus, isminus]
            else:
                raise ValueError(f"Unhandled merge mode: {split!r}")
            for j, g in enumerate(groups):
                logger.info("Merging group {}...", j)
                Imj, sigmamj, countsj = S.merge(mask=S.mask | ~g)
                cols[f"group_{j}_intensity"] = Imj.astype(np.float32)
                cols[f"group_{j}_intensity_error"] = sigmamj.astype(np.float32)
                cols[f"group_{j}_count"] = countsj.astype(np.int32)

    # create the output table object
    hkl_table = HKLTable(h, k, l, ndiv=hkl.ndiv, **cols)
    logger.info("Unique reflections in merged output: {}", len(hkl_table))

    logger.info("Saving merged data to {}...", params.outfile)
    with P.step("save"):
        saveobj(hkl_table, params.outfile, name="hkl_table", append=False)
    P.summary()
    logger.info("Merge completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_merge))


if __name__ == "__main__":
    run()
