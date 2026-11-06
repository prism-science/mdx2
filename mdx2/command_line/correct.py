"""
Apply corrections to integrated data
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from scipy.ndimage import map_coordinates
from simple_parsing import field  # pip install simple-parsing

from mdx2 import profiling
from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.integration import log_timers
from mdx2.io import loadobj, saveobj
from mdx2.utils import _fraction


@dataclass
class Parameters:
    """Options for applying corrections to integrated data"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    hkl: str = field(positional=True)  # NeXus data file containing hkl_table
    background: Optional[str] = None  # NeXus file with background map
    attenuation: bool = True  # apply attenuation correction
    efficiency: bool = True  # apply efficiency correction
    polarization: bool = True  # apply polarization correction
    lorentz: bool = False  # apply Lorentz correction
    p1: bool = False  # map Miller indices to asymmetric unit for P1 (Friedel symmetry only)
    nproc: int = 1  # ignored (mdx2.correct runs in one process; see nthreads)
    nthreads: int = 4  # threads (row blocks in parallel); each adds ~0.15 GB of temporaries (1 M-row blocks)
    engine: str = field(default="fast", choices=["fast", "reference"])
    """fast: row blocks in --nthreads threads, compiled asymmetric-unit mapping, results written into the output
    columns (no full copy of the table); identical output. reference: mdx2 <= 1.0.5 (kept for one release)"""
    profile: bool = False  # log time per step (summed over the blocks)
    outfile: str = "corrected.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.correct.log"  # name of the log file


BLOCK_ROWS = 1 << 20  # rows per block (bounded temporaries; the unit of work for the threads)


def correct_block(T, sl, ctx, out):
    """Corrections for the rows sl of the integrated table T, written into the output columns out (same arithmetic,
    in the same order, as run_correct_reference). Returns step times."""
    t = {}
    clock = time.perf_counter
    t0 = clock()
    h, k, l = T.h[sl], T.k[sl], T.l[sl]  # noqa: E741
    s = ctx["UB"] @ np.stack((h, k, l))
    out["s"][sl] = np.sqrt(np.sum(s * s, axis=0))
    t1 = clock()
    ndiv = T.ndiv
    H = np.round(h * ndiv[0]).astype(int)
    K = np.round(k * ndiv[1]).astype(int)
    L = np.round(l * ndiv[2]).astype(int)
    if ctx["symmetry"] is None:
        op = (L < 0).astype(int)
        L = np.abs(L)
    else:
        H, K, L, op = ctx["symmetry"].to_asu(H, K, L)
    out["h"][sl] = H / ndiv[0]
    out["k"][sl] = K / ndiv[1]
    out["l"][sl] = L / ndiv[2]
    out["op"][sl] = op
    t2 = clock()
    iy, ix = T.iy[sl], T.ix[sl]
    coords = np.stack((_fraction(iy, axis=ctx["c_iy"]), _fraction(ix, axis=ctx["c_ix"])))
    corr = {key: map_coordinates(m, coords, order=1) for key, m in ctx["maps"].items()}
    t3 = clock()
    count_rate = T.counts[sl] / T.seconds[sl]
    count_rate_error = np.sqrt(T.counts[sl]) / T.seconds[sl]
    if ctx["background"] is not None:
        count_rate = count_rate - ctx["background"].interpolate(T.phi[sl], iy, ix)
    t4 = clock()
    solid_angle = corr["solid_angle"].copy()
    for name in ctx["applied"]:
        solid_angle *= corr[name]
    rs_volume = T.pixels[sl] * corr["d3s"] / ctx["det_UB"]
    intensity = count_rate / solid_angle
    intensity_error = count_rate_error / solid_angle
    if ctx["lorentz"]:
        intensity *= rs_volume
        intensity_error *= rs_volume
    out["rs_volume"][sl] = rs_volume
    out["intensity"][sl] = intensity
    out["intensity_error"][sl] = intensity_error
    for name in ("ix", "iy", "phi", "n"):
        out[name][sl] = getattr(T, name)[sl]
    t5 = clock()
    t.update({"|s|": t1 - t0, "to_asu": t2 - t1, "corrections": t3 - t2, "background": t4 - t3, "intensities": t5 - t4})
    return t


def run_correct(params):
    """Run the correct script"""
    if params.engine == "reference":
        return run_correct_reference(params)
    if params.nproc != 1:
        logger.warning("--nproc is ignored: mdx2.correct uses --nthreads threads in one process")
    P = profiling.PROFILER = profiling.Profiler(params.profile)
    clock = time.perf_counter

    t0 = clock()
    logger.info("Loading integrated data and geometry...")
    T = loadobj(params.hkl, "hkl_table")
    if "_ndiv" in T.__dict__:  # hack to work with older versions
        T.ndiv = T._ndiv
        del T._ndiv
    Corrections = loadobj(params.geom, "corrections")
    Crystal = loadobj(params.geom, "crystal")
    if params.p1:
        logger.info("Using P1 symmetry only (ignoring space group)")
        Symmetry = None
    else:
        Symmetry = loadobj(params.geom, "symmetry")
    Bkg = loadobj(params.background, "binned_image_series") if params.background is not None else None
    P.add("load", clock() - t0)

    UB = Crystal.ub_matrix
    stacked = Corrections.data  # (iy, ix, 6), as Corrections.interpolate uses it
    applied = [n for n in ("attenuation", "efficiency", "polarization") if getattr(params, n)]
    ctx = dict(
        UB=UB, det_UB=np.linalg.det(UB), symmetry=Symmetry, background=Bkg, lorentz=params.lorentz, applied=applied,
        c_iy=Corrections.iy, c_ix=Corrections.ix,
        maps={key: stacked[..., i] for i, key in enumerate(Corrections._data_keys)},
    )  # fmt: skip
    if applied:
        logger.info("Applied corrections: {}", ", ".join(applied))
    if Bkg is not None:
        logger.info("Subtracting background from count rate")
    if params.lorentz:
        logger.info("Applying Lorentz correction")

    n = len(T)
    out = {c: np.empty(n, dtype=np.float32) for c in ("s", "h", "k", "l", "rs_volume", "intensity", "intensity_error")}
    out.update({c: np.empty(n, dtype=np.float32) for c in ("ix", "iy", "phi")})
    out.update({c: np.empty(n, dtype=np.int32) for c in ("op", "n")})
    blocks = [slice(i, min(i + BLOCK_ROWS, n)) for i in range(0, n, BLOCK_ROWS)]
    logger.info(
        "Correcting {} rows: asymmetric unit, corrections{}, in {} blocks ({} threads)...",
        n, " and background" if Bkg is not None else "", len(blocks), params.nthreads,
    )  # fmt: skip
    t0 = clock()
    with ThreadPoolExecutor(max(1, params.nthreads)) as pool:
        timers = list(pool.map(lambda sl: correct_block(T, sl, ctx, out), blocks))
    wall = clock() - t0
    P.add("blocks", wall)
    log_timers(timers, wall, params.profile)

    # the output table: as run_correct_reference (same columns, same order)
    t0 = clock()
    T.s = out.pop("s")
    T.h, T.k, T.l = out.pop("h"), out.pop("k"), out.pop("l")
    T.op = out.pop("op")
    T.rs_volume = out.pop("rs_volume")
    T.intensity = out.pop("intensity")
    T.intensity_error = out.pop("intensity_error")
    del T.counts
    del T.seconds
    del T.pixels
    T.ix, T.iy, T.phi, T.n = out.pop("ix"), out.pop("iy"), out.pop("phi"), out.pop("n")
    logger.info("Reflections processed: {}", len(T))
    logger.info("Saving corrected data to {}...", params.outfile)
    saveobj(T, params.outfile, name="hkl_table", append=False)
    P.add("save", clock() - t0)
    P.summary()
    logger.info("Corrections completed successfully")


def run_correct_reference(params):
    """mdx2 <= 1.0.5 (whole-table operations, single-threaded); kept for one release."""
    hkl = params.hkl
    geom = params.geom
    background = params.background
    outfile = params.outfile
    p1 = params.p1
    attenuation = params.attenuation
    efficiency = params.efficiency
    polarization = params.polarization
    lorentz = params.lorentz

    logger.info("Loading integrated data and geometry...")
    T = loadobj(hkl, "hkl_table")

    # hack to work with older versions
    if "_ndiv" in T.__dict__:
        T.ndiv = T._ndiv
        del T._ndiv

    Corrections = loadobj(geom, "corrections")
    Crystal = loadobj(geom, "crystal")

    if p1:
        logger.info("Using P1 symmetry only (ignoring space group)")
        Symmetry = None
    else:
        Symmetry = loadobj(geom, "symmetry")

    UB = Crystal.ub_matrix

    logger.info("Calculating scattering vector magnitudes...")
    s = UB @ np.stack((T.h, T.k, T.l))
    T.s = np.sqrt(np.sum(s * s, axis=0))

    logger.info("Mapping Miller indices to asymmetric unit...")
    T = T.to_asu(Symmetry)

    # apply corrections to intensities
    logger.info("Interpolating correction factors...")
    Cinterp = Corrections.interpolate(T.iy, T.ix)

    count_rate = T.counts / T.seconds
    count_rate_error = np.sqrt(T.counts) / T.seconds

    if background is not None:
        Bkg = loadobj(background, "binned_image_series")
        bkg_count_rate = Bkg.interpolate(T.phi, T.iy, T.ix)
        logger.info("Subtracting background from count rate")
        count_rate = count_rate - bkg_count_rate

    solid_angle = Cinterp["solid_angle"].copy()

    corrections_applied = []
    if attenuation:
        solid_angle *= Cinterp["attenuation"]
        corrections_applied.append("attenuation")
    if efficiency:
        solid_angle *= Cinterp["efficiency"]
        corrections_applied.append("efficiency")
    if polarization:
        solid_angle *= Cinterp["polarization"]
        corrections_applied.append("polarization")

    if corrections_applied:
        logger.info("Applied corrections: {}", ", ".join(corrections_applied))

    logger.info("Computing reciprocal space volume fractions...")
    T.rs_volume = T.pixels * Cinterp["d3s"] / np.linalg.det(UB)

    logger.info("Computing intensities and errors...")
    T.intensity = count_rate / solid_angle
    T.intensity_error = count_rate_error / solid_angle

    if lorentz:
        logger.info("Applying Lorentz correction")
        T.intensity *= T.rs_volume
        T.intensity_error *= T.rs_volume

    # remove some unnecessary columns
    del T.counts
    del T.seconds
    del T.pixels

    # save some disk space
    T.h = T.h.astype(np.float32)
    T.k = T.k.astype(np.float32)
    T.l = T.l.astype(np.float32)
    T.s = T.s.astype(np.float32)
    T.intensity = T.intensity.astype(np.float32)
    T.intensity_error = T.intensity_error.astype(np.float32)
    T.ix = T.ix.astype(np.float32)
    T.iy = T.iy.astype(np.float32)
    T.phi = T.phi.astype(np.float32)
    T.rs_volume = T.rs_volume.astype(np.float32)
    T.n = T.n.astype(np.int32)
    T.op = T.op.astype(np.int32)

    logger.info("Reflections processed: {}", len(T))
    logger.info("Saving corrected data to {}...", outfile)
    saveobj(T, outfile, name="hkl_table", append=False)
    logger.info("Corrections completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_correct))


if __name__ == "__main__":
    run()
