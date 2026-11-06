"""
Find and analyze peaks in an image stack
"""

import time
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field  # pip install simple-parsing

from mdx2 import profiling
from mdx2._integration_kernels import threshold_indices_kernel
from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.data import Peaks
from mdx2.geometry import GaussianPeak
from mdx2.integration import chunk_columns, log_timers
from mdx2.io import loadobj, saveobj
from mdx2.parallel import available_cpus


@dataclass
class Parameters:
    """Options for finding peaks in an image stack"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    data: str = field(positional=True)  # NeXus data file containing image_series
    count_threshold: float  # pixels with counts above threshold are flagged as peaks
    sigma_cutoff: float = 3.0  # for outlier rejection in Gaussian peak fitting
    outfile: str = "peaks.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.find_peaks.log"  # name of the log file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)
    engine: str = "fast"
    """fast: one task per chunk column (detector module), pixels found with a compiled kernel; the Miller-index grid is
    loaded only after the search, to index the peak pixels (same peaks, in the same order, as reference). reference:
    mdx2 <= 1.0.5 (one task per chunk; kept for one release)"""
    profile: bool = False  # log time per step, summed over the workers (fast engine)

    def __post_init__(self):
        """Validate sigma_cutoff parameter"""
        if self.sigma_cutoff <= 0:
            raise ValueError(f"sigma_cutoff must be > 0, got {self.sigma_cutoff}")
        if self.engine not in ("fast", "reference"):
            raise ValueError(f"Unknown engine {self.engine!r} (use 'fast' or 'reference')")


def find_peaks_column(image_series, slices, count_threshold):
    """Peak pixels of one chunk column: per chunk with peaks, (phi start, (phi, iy, ix)), plus step times."""
    timer = {"read": 0.0, "threshold": 0.0}
    found = []
    for sl in slices:
        t0 = time.perf_counter()
        ims = image_series[sl]
        data = np.ascontiguousarray(ims.data)
        t1 = time.perf_counter()
        flat = threshold_indices_kernel(data, float(count_threshold), ims._maskval)
        timer["read"] += t1 - t0
        if flat.size:
            z, y, x = np.unravel_index(flat, data.shape)
            found.append((sl[0].start, (ims.phi[z], ims.iy[y], ims.ix[x])))
        timer["threshold"] += time.perf_counter() - t1
    return found, timer


def run_find_peaks(params):
    """Run the find peaks script"""
    geom = params.geom
    data = params.data
    count_threshold = params.count_threshold
    sigma_cutoff = params.sigma_cutoff
    outfile = params.outfile
    nproc = params.nproc

    P_ = profiling.PROFILER = profiling.Profiler(params.profile)
    clock = time.perf_counter
    t0 = clock()
    logger.info("Loading geometry and image data...")
    # fast engine: the Miller-index grid is loaded after the search (only the main process needs it, for indexing)
    MI = loadobj(geom, "miller_index") if params.engine == "reference" else None
    IS = loadobj(data, "image_series")
    P_.add("load", clock() - t0)

    logger.info("Finding pixels above threshold: {}", count_threshold)

    if params.engine == "reference":
        # Find peaks in parallel
        def peaksearch(sl):
            ims = IS[sl]
            im_data = ims.data_masked
            peaks = Peaks.where(im_data > count_threshold, ims.phi, ims.iy, ims.ix)
            if peaks.size:
                return peaks

        slices = list(IS.chunk_slice_iterator())
        logger.info("Searching for peaks in {} image chunks (requested n_jobs: {})...", len(slices), nproc)
        t0 = clock()
        with Parallel(n_jobs=nproc, verbose=10) as parallel:
            log_parallel_backend(parallel)
            peaklist = parallel(delayed(peaksearch)(sl) for sl in slices)
        P_.add("tasks", clock() - t0)
        logger.info("Peak search completed")

        peaks_nonempty = [p for p in peaklist if p is not None]
        if not peaks_nonempty:
            raise ValueError(
                f"No peaks found above threshold {count_threshold}. Try lowering the threshold or check your data."
            )
        P = Peaks.stack(peaks_nonempty)
        logger.info("Found {} peak pixels", P.size)

        logger.info("Indexing peaks...")
        t0 = clock()
        h, k, l = MI.interpolate(P.phi, P.iy, P.ix)  # noqa: E741
        P_.add("index (main)", clock() - t0)
    else:
        columns = chunk_columns(IS)
        logger.info(
            "Searching for peaks in {} chunk columns (detector modules) of {} chunks each (requested n_jobs: {})...",
            len(columns), len(columns[0]), nproc,
        )  # fmt: skip
        if nproc > len(columns) or (nproc < 0 and available_cpus() > len(columns)):
            logger.info("At most {} processes are used (one per chunk column)", len(columns))

        def column_task(slices):
            return find_peaks_column(IS, slices, count_threshold)

        t0 = clock()
        with Parallel(n_jobs=nproc, verbose=10) as parallel:
            log_parallel_backend(parallel)
            results = parallel(delayed(column_task)(c) for c in columns)
        wall = clock() - t0
        P_.add("tasks", wall)
        log_timers([r[1] for r in results], wall, params.profile)
        logger.info("Peak search completed")

        # the reference engine's order: chunks as chunk_slice_iterator (phi, then iy, then ix), pixels in C order
        t0 = clock()
        order = {(sl[0].start, sl[1].start, sl[2].start): i for i, sl in enumerate(IS.chunk_slice_iterator())}
        parts = []
        for c, (found, _) in zip(columns, results):
            starts = {sl[0].start: (sl[1].start, sl[2].start) for sl in c}
            parts += [(order[(z0, *starts[z0])], arrays) for z0, arrays in found]
        if not parts:
            raise ValueError(
                f"No peaks found above threshold {count_threshold}. Try lowering the threshold or check your data."
            )
        parts.sort(key=lambda p: p[0])
        P = Peaks(*(np.concatenate([p[1][i] for p in parts]) for i in range(3)))
        del parts, results
        P_.add("concatenate", clock() - t0)
        logger.info("Found {} peak pixels", P.size)

        logger.info("Indexing peaks...")
        t0 = clock()
        MI = loadobj(geom, "miller_index")
        P_.add("load geometry", clock() - t0)
        t0 = clock()
        h, k, l = MI.interpolate(P.phi, P.iy, P.ix)  # noqa: E741
        del MI
        P_.add("index (main)", clock() - t0)

    dh = h - np.round(h)
    dk = k - np.round(k)
    dl = l - np.round(l)

    logger.info("Fitting Gaussian peak model...")
    t0 = clock()
    GP, is_outlier = GaussianPeak.fit_to_points(dh, dk, dl, sigma_cutoff=sigma_cutoff)
    P_.add("fit", clock() - t0)
    logger.info("Rejected {} outliers (sigma cutoff: {})", np.sum(is_outlier), sigma_cutoff)
    logger.info("Peak model r0: {}", GP.r0)

    # Compute principal axes of error ellipsoid using SVD
    # sigma = U @ diag(s) @ V.T, where U contains the principal axes
    # and s contains the semi-axis lengths
    U, s, _Vt = np.linalg.svd(GP.sigma)

    logger.info("Error ellipsoid semi-axis lengths: {}", s)
    logger.info("Error ellipsoid principal axis 1: {}", U[:, 0])
    logger.info("Error ellipsoid principal axis 2: {}", U[:, 1])
    logger.info("Error ellipsoid principal axis 3: {}", U[:, 2])

    Outliers = Peaks(P.phi[is_outlier], P.iy[is_outlier], P.ix[is_outlier])

    logger.info("Saving peaks to {}...", outfile)
    t0 = clock()
    saveobj(GP, outfile, name="peak_model", append=False)
    saveobj(P, outfile, name="peaks", append=True)
    saveobj(Outliers, outfile, name="outliers", append=True)
    P_.add("save", clock() - t0)
    P_.summary()
    logger.info("Peak finding completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_find_peaks))

if __name__ == "__main__":
    run()
