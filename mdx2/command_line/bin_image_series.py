"""
Bin down an image stack
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field

from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.geometry import GridData
from mdx2.io import loadobj, nxload, saveobj
from mdx2.utils import slice_sections


@dataclass
class Parameters:
    """Options for binning a series of images in the phi and xy directions"""

    data: str = field(positional=True)  # NeXus data file containing the image_series
    bins: Tuple[int, int, int] = field(positional=True)  # number per bin in each direction (frames, y, x)
    mask: Optional[str] = None  # name of NeXus file containing the mask
    valid_range: Optional[Tuple[int, int]] = None  # minimum and maximum valid data values
    outfile: str = "binned.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.bin_image_series.log"  # name of the log file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)

    def __post_init__(self):
        """Validate bins and valid_range parameters"""
        for i, bin_size in enumerate(self.bins):
            if bin_size <= 0:
                raise ValueError(f"bins[{i}] must be positive, got {bin_size}")
        if self.valid_range is not None and self.valid_range[0] >= self.valid_range[1]:
            raise ValueError(f"valid_range[0] must be < valid_range[1], got {self.valid_range}")


def bin_slab(data, maskval, valid_range, mask, y_starts, x_starts):
    """Mean of the valid pixels of a slab of frames in each (y, x) bin; NaN where a bin has none. Valid: not maskval,
    within valid_range (inclusive) if given, and not set in mask (True = masked) if given. Bins start at y_starts /
    x_starts. Integer data are summed in int64 and divided once, as np.ma.mean does, so the means are identical to
    the masked-array version this replaces (a Python loop over bins; ~3x slower)."""
    valid = data != maskval
    if valid_range is not None:
        valid &= (data >= valid_range[0]) & (data <= valid_range[1])
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    acc = np.int64 if np.issubdtype(data.dtype, np.integer) else np.float64
    sums = np.where(valid, data, 0).sum(axis=0, dtype=acc)
    counts = valid.sum(axis=0, dtype=np.int64)
    sums = np.add.reduceat(np.add.reduceat(sums, y_starts, axis=0), x_starts, axis=1)
    counts = np.add.reduceat(np.add.reduceat(counts, y_starts, axis=0), x_starts, axis=1)
    out = np.full(sums.shape, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def run_bin_image_series(params):
    """Run the binning algorithm"""
    data = params.data
    bins = params.bins
    outfile = params.outfile
    valid_range = params.valid_range
    nproc = params.nproc
    maskfile = params.mask

    logger.info("Loading image series...")
    image_series = loadobj(data, "image_series")
    logger.info("Original image shape: {}", image_series.shape)

    if maskfile is not None:
        logger.info("Loading mask...")
        # Use nxload directly instead of loadobj because loadobj fails for very large arrays
        nxs = nxload(maskfile)
        if nxs.entry.mask.attrs.get("mdx2_class") == "PeakMask":
            raise ValueError(
                f"{maskfile} is a dynamic peak mask (peak model + pixel list), which bin_image_series does not support "
                "(it needs the Miller-index grid); write a static mask with mdx2.mask_peaks --static"
            )
        mask = nxs.entry.mask.signal  # nxfield
    else:
        mask = None

    logger.info("Binning images with bin size: {}", bins)

    # Compute binning parameters
    bins_array = np.array(bins)
    nbins = np.ceil(image_series.shape / bins_array).astype(int)
    sl_0 = slice_sections(image_series.shape[0], nbins[0])
    sl_1 = slice_sections(image_series.shape[1], nbins[1])
    sl_2 = slice_sections(image_series.shape[2], nbins[2])

    new_phi = np.array([image_series.phi[sl].mean() for sl in sl_0])
    new_iy = np.array([image_series.iy[sl].mean() for sl in sl_1])
    new_ix = np.array([image_series.ix[sl].mean() for sl in sl_2])

    y_starts = np.array([sl.start for sl in sl_1])
    x_starts = np.array([sl.start for sl in sl_2])

    def binslab(sl):
        data = image_series._as_np(image_series.data[sl, :, :])
        msk = image_series._as_np(mask[sl, :, :]) if mask is not None else None
        return bin_slab(data, image_series._maskval, valid_range, msk, y_starts, x_starts)

    logger.info("Binning {} image slabs (requested n_jobs: {})...", len(sl_0), nproc)
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        log_parallel_backend(parallel)
        new_data = np.stack(parallel(delayed(binslab)(sl) for sl in sl_0))
    logger.info("Binning computation completed")

    # Convert to count rate
    new_times = np.array([image_series.exposure_times[sl].mean() for sl in sl_0])
    new_data = new_data / new_times[:, np.newaxis, np.newaxis]

    binned = GridData((new_phi, new_iy, new_ix), new_data, axes_names=["phi", "iy", "ix"])
    logger.info("Binned image shape: {}", binned.data.shape)

    logger.info("Saving binned data to {}...", outfile)
    saveobj(binned, outfile, name="binned_image_series")
    logger.info("Binning completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_bin_image_series))


if __name__ == "__main__":
    run()
