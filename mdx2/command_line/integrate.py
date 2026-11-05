"""
Integrate counts in an image stack on a Miller index grid
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field

from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.data import HKLTable
from mdx2.geometry import load_mask
from mdx2.integration import bin_chunk, chunk_columns, log_timers
from mdx2.io import (
    loadobj,
    saveobj,
)
from mdx2.parallel import available_cpus


@dataclass
class Parameters:
    """Options for integrating counts on a Miller index grid"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    data: str = field(positional=True)  # NeXus data file containing image_series
    mask: Optional[str] = None  # NeXus data file containing mask
    subdivide: Tuple[int, int, int] = (1, 1, 1)  # subdivisions of the Miller index grid
    max_spread: float = 1.0  # maximum angular spread (degrees) for binning partial observations
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)
    outfile: str = "integrated.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.integrate.log"  # name of the log file
    engine: str = "fast"
    """"fast" (default): one task per column of chunks along phi (a detector module with the default chunking),
    compiled per-chunk binning, partial observations grouped (max_spread) within the module, so observations never
    combine pixels from different modules; "reference": mdx2 <= 1.0.5 (one task per chunk, grouping over the whole
    detector; kept for one release)"""
    profile: bool = False  # log time per step, summed over the workers (fast engine)

    def __post_init__(self):
        """Validate subdivide and max_spread parameters"""
        for i, div in enumerate(self.subdivide):
            if div <= 0:
                raise ValueError(f"subdivide[{i}] must be > 0, got {div}")
        if self.max_spread <= 0:
            raise ValueError(f"max_spread must be > 0, got {self.max_spread}")
        if self.engine not in ("fast", "reference"):
            raise ValueError(f"Unknown engine {self.engine!r} (use 'fast' or 'reference')")


def group_partials(tabs, max_degrees):
    """Group per-chunk partial sums into observations: partials of a voxel whose mean phi lies within the same
    max_degrees interval, counted from the voxel's earliest partial, are summed; mean positions. DataFrame indexed by
    (h, k, l, n)."""
    df = HKLTable.concatenate(tabs).to_frame()
    df["tmp"] = df["phi"] / df["pixels"]
    delta_phi = df.tmp - df.groupby(["h", "k", "l"])["tmp"].transform("min")
    df["n"] = np.floor(delta_phi / max_degrees)
    df = df.drop(columns=["tmp"])
    df = df.groupby(["h", "k", "l", "n"]).sum()
    df["phi"] = df["phi"] / df["pixels"]
    df["iy"] = df["iy"] / df["pixels"]
    df["ix"] = df["ix"] / df["pixels"]
    return df


def integrate_column(image_series, slices, miller_index, mask, ndiv, max_degrees):
    """Fast-engine task: bin every chunk of one column (one detector module), group within the column.
    Returns (DataFrame or None, number of partial rows, seconds per step)."""
    timer = {}
    tabs = [t for t in (bin_chunk(image_series, sl, miller_index, mask, ndiv, timer=timer) for sl in slices) if t]
    if not tabs:
        return None, 0, timer
    t0 = time.perf_counter()
    df = group_partials(tabs, max_degrees)
    timer["group"] = time.perf_counter() - t0
    return df, sum(len(t) for t in tabs), timer


def run_integrate(params):
    """Run the integrate script"""
    geom = params.geom
    data = params.data
    outfile = params.outfile
    nproc = params.nproc
    ndiv = params.subdivide
    max_degrees = params.max_spread
    maskfile = params.mask

    logger.info("Loading geometry and image data...")
    MI = loadobj(geom, "miller_index")
    IS = loadobj(data, "image_series")

    if maskfile is not None:
        logger.info("Loading mask...")
        mask = load_mask(maskfile)  # static (read per chunk) or dynamic (rebuilt per chunk)
        mask.check_axes(IS.phi, IS.iy, IS.ix)
        logger.info("Mask type: {}", type(mask).__name__)
    else:
        mask = None

    if params.engine == "reference":

        def intchunk(sl):
            ims = IS[sl]
            tab = ims.index(MI, mask=None if mask is None else mask.for_chunk(sl))
            tab.ndiv = ndiv
            return tab.bin(count_name="pixels")

        slices = list(IS.chunk_slice_iterator())
        logger.info("Integrating {} image chunks (requested n_jobs: {})...", len(slices), nproc)
        with Parallel(n_jobs=nproc, verbose=10) as parallel:
            log_parallel_backend(parallel)
            T = parallel(delayed(intchunk)(sl) for sl in slices)
        logger.info("Integration completed")
        logger.info("Summing partial observations over {} chunks...", len(T))
        if not T:
            raise ValueError(
                "No integration results produced. The image series may be empty, or no valid data was found to "
                "integrate."
            )
        df = group_partials(T, max_degrees)
        voxels_before = np.sum([len(t) for t in T])
    else:
        columns = chunk_columns(IS)
        logger.info(
            "Integrating {} chunk columns (detector modules) of {} chunks each (requested n_jobs: {})...",
            len(columns), len(columns[0]), nproc,
        )  # fmt: skip
        if nproc > len(columns) or (nproc < 0 and available_cpus() > len(columns)):
            logger.info("At most {} processes are used (one per chunk column)", len(columns))
        t0 = time.perf_counter()

        def intcolumn(slices):
            return integrate_column(IS, slices, MI, mask, ndiv, max_degrees)

        with Parallel(n_jobs=nproc, verbose=10) as parallel:
            log_parallel_backend(parallel)
            results = parallel(delayed(intcolumn)(c) for c in columns)
        wall = time.perf_counter() - t0
        logger.info("Integration completed")
        frames = [r[0] for r in results if r[0] is not None]
        if not frames:
            raise ValueError(
                "No integration results produced. The image series may be empty, or no valid data was found to "
                "integrate."
            )
        log_timers([r[2] for r in results], wall, params.profile)
        df = pd.concat(frames)
        voxels_before = sum(r[1] for r in results)

    voxels_after = len(df)
    logger.info("Binned from {} to {} voxels", voxels_before, voxels_after)

    hkl_table = HKLTable.from_frame(df)
    hkl_table.ndiv = ndiv  # lost in conversion to/from dataframe

    hkl_table.h = hkl_table.h.astype(np.float32)
    hkl_table.k = hkl_table.k.astype(np.float32)
    hkl_table.l = hkl_table.l.astype(np.float32)
    hkl_table.phi = hkl_table.phi.astype(np.float32)
    hkl_table.ix = hkl_table.ix.astype(np.float32)
    hkl_table.iy = hkl_table.iy.astype(np.float32)
    hkl_table.seconds = hkl_table.seconds.astype(np.float32)
    hkl_table.counts = hkl_table.counts.astype(np.int32)
    hkl_table.pixels = hkl_table.pixels.astype(np.int32)

    logger.info("Saving integrated data to {}...", outfile)
    saveobj(hkl_table, outfile, name="hkl_table", append=False)
    logger.info("Integration completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_integrate))


if __name__ == "__main__":
    run()
