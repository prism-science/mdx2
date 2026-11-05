"""
Import x-ray image data using the dxtbx machinery
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field

from mdx2._import_kernels import OVERFLOW16, pack_frame_kernel, pack_raw_kernel, unpack_kernel
from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.data import ImageSeries
from mdx2.dxtbx_machinery import ImageSet


@dataclass
class Parameters:
    """Options for importing x-ray image data"""

    expt: str = field(positional=True)  # experiments file, such as from dials.import
    chunks: Optional[Tuple[int, int, int]] = field(
        default=None, help="chunking for compression (frames, y, x). Use -1 for default"
    )
    outfile: str = "data.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.import_data.log"  # name of the log file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)
    datastore: str = "datastore"  # folder for storing source datasets
    compression: str = field(default="blosc-lz4", choices=["blosc-lz4", "blosc-zstd", "gzip"])
    """datastore compression: blosc-lz4 (default; fast to read and write), blosc-zstd (~20% smaller, slower to
    write), gzip (gzip-1 + shuffle, as mdx2 <= 1.0.5). Blosc needs the hdf5plugin filters wherever the data is read."""
    reader: str = field(default="auto", choices=["auto", "dxtbx"])
    """auto: read NXmx (e.g. Eiger) files directly with h5py and apply the dxtbx mask in numpy, ~2.5x faster (the first
    frame is checked against dxtbx; if it differs, dxtbx is used); other formats are read with dxtbx.
    dxtbx: always read with dxtbx."""


# ---- one batch of frames (a datastore file), in a worker -------------------------------------------------------------


def pack_frame(buf, frame, i, overflow):
    """Store an int32 frame (masked = -1) in buf (uint16) exactly: values 0..65533 as is, -1 as MASKED16, anything
    else as OVERFLOW16 with (frame index, y, x, value) appended to overflow."""
    if pack_frame_kernel(frame, buf):
        y, x = np.nonzero(buf == OVERFLOW16)
        overflow.append((np.full(y.size, i, dtype=np.int64), y, x, frame[y, x]))


def pack_raw(buf, reader, raw, i, overflow):
    """As pack_frame, from a raw detector frame of a NXmxFrameReader (mask applied in the same pass)."""
    valid = reader.static_mask if reader.static_mask is not None else np.ones((1, 1), dtype=np.bool_)
    if pack_raw_kernel(raw, reader.lo, reader.hi, valid, reader.static_mask is not None, buf):
        y, x = np.nonzero(buf == OVERFLOW16)
        overflow.append((np.full(y.size, i, dtype=np.int64), y, x, raw[y, x].astype(np.int32)))


def unpack(codes, overflow_arrays, z0=0, y0=0, x0=0):
    """int32 values of a block of the uint16 buffer (block origin z0, y0, x0 in the batch)."""
    out = np.empty(codes.shape, dtype=np.int32)
    unpack_kernel(codes, out)
    if overflow_arrays is not None:
        f, y, x, v = overflow_arrays
        nz, ny, nx = codes.shape
        sel = (f >= z0) & (f < z0 + nz) & (y >= y0) & (y < y0 + ny) & (x >= x0) & (x < x0 + nx)
        out[f[sel] - z0, y[sel] - y0, x[sel] - x0] = v[sel]
    return out


def write_chunks(dataset, buf, overflow, chunks):
    """Write the uint16 assembly buffer to an int32 HDF5 dataset one chunk column at a time (restoring -1 and the
    overflow values), so that no full-size int32 copy is made."""
    ov = tuple(np.concatenate(a) for a in zip(*overflow)) if overflow else None
    n, ny, nx = buf.shape
    for y0 in range(0, ny, chunks[1]):
        y1 = min(y0 + chunks[1], ny)
        for x0 in range(0, nx, chunks[2]):
            x1 = min(x0 + chunks[2], nx)
            dataset[:, y0:y1, x0:x1] = unpack(buf[:, y0:y1, x0:x1], ov, 0, y0, x0)


MAX_OVERFLOW_FRACTION = 0.01  # beyond this, a batch is assembled as int32 instead (the overflow list would be larger)


def import_batch(iset, reader, istart, istop, filename, datapath):
    """Read frames istart..istop-1 into a uint16 buffer (+ overflow list) and write them to the datastore file.
    Frames come from reader (a NXmxFrameReader; h5py only) if given, else from iset (dxtbx).

    Memory: 2 bytes per pixel for the batch, plus one frame and one chunk column as int32 (the previous version held the
    batch twice as int32: 8 bytes per pixel). If more than MAX_OVERFLOW_FRACTION of the pixels exceed 65533, the batch
    is assembled as int32 instead (4 bytes per pixel). Returns step times and the number of overflow pixels."""
    T = {"read": 0.0, "pack": 0.0, "write": 0.0}
    n = istop - istart
    ny, nx = iset.frame_shape if reader is None else reader.frame_shape
    buf = np.empty((n, ny, nx), dtype=np.uint16)
    big = None  # int32 batch, if the uint16 buffer was abandoned
    overflow, noverflow = [], 0
    for i, ind in enumerate(range(istart, istop)):
        t0 = time.perf_counter()
        if big is not None:
            if reader is None:
                big[i] = iset.read_frame(ind)
            else:
                reader.read_frame(ind, out=big[i])
            T["read"] += time.perf_counter() - t0
            continue
        if reader is None:
            frame = iset.read_frame(ind)
            t1 = time.perf_counter()
            pack_frame(buf[i], frame, i, overflow)
            T["pack"] += time.perf_counter() - t1
        else:
            raw = reader.read_raw(ind)
            t1 = time.perf_counter()
            pack_raw(buf[i], reader, raw, i, overflow)
            T["pack"] += time.perf_counter() - t1
        T["read"] += t1 - t0
        noverflow = sum(o[0].size for o in overflow)
        if noverflow > MAX_OVERFLOW_FRACTION * buf.size:  # switch this batch to int32
            ov = tuple(np.concatenate(a) for a in zip(*overflow))
            big = np.empty((n, ny, nx), dtype=np.int32)
            big[: i + 1] = unpack(buf[: i + 1], ov)
            del buf, ov
            overflow = []
    t3 = time.perf_counter()
    with h5py.File(filename, "r+") as fh:
        dataset = fh[datapath]
        if big is None:
            write_chunks(dataset, buf, overflow, dataset.chunks)
        else:
            dataset[:, :, :] = big
    T["write"] += time.perf_counter() - t3
    return T, (int(np.sum(big > OVERFLOW16 - 1)) if big is not None else noverflow)


def run_import_data(params):
    exptfile = params.expt
    chunks = params.chunks
    nproc = params.nproc
    outfile = params.outfile
    datastore = params.datastore

    logger.info("Loading experiment metadata...")
    image_series = ImageSeries.from_expt(exptfile, compression=params.compression)
    logger.info("Datastore compression: {}", params.compression)
    iset = ImageSet.from_file(exptfile)
    logger.info("Image data shape (phi, iy, ix): {}", image_series.shape)
    reader = None
    if params.reader == "auto":
        reader, why = iset.direct_reader()
        if reader is None:
            logger.info("Reading frames with dxtbx ({})", why)
        else:
            mismatch = reader.check_against(iset)
            reader.close()
            if mismatch is None:
                logger.info("Reading frames directly from the NXmx data files (first frame identical to dxtbx)")
            else:
                logger.warning("Direct NXmx reader differs from dxtbx ({}); reading frames with dxtbx", mismatch)
                reader = None
    else:
        logger.info("Reading frames with dxtbx")

    if chunks is not None:
        # override the default chunking
        # if any of the chunks dimensions is <=0, use the default for that dimension
        default_chunks = image_series.data.chunks
        chunks = tuple(c if c > 0 else d for c, d in zip(chunks, default_chunks))
        image_series.data.chunks = chunks
        logger.info("Using chunking: {}", chunks)

    logger.info("Creating virtual dataset structure...")
    image_series.save(
        outfile,
        virtual=True,
        source_directory=datastore,
    )

    slices = [sl for sl in image_series.chunk_slice_along_axis(0)]

    # Access virtual dataset information through public API
    # Reload the ImageSeries to ensure we have the updated virtual dataset
    image_series_reloaded = ImageSeries.load(outfile)
    files = image_series_reloaded.virtual_source_files
    vpath = image_series_reloaded.virtual_dataset_path

    # These edge cases should not happen if virtual datasets are implemented correctly, but check anyway
    if len(files) != len(slices):
        raise RuntimeError(f"Virtual dataset mismatch: {len(slices)=} vs {len(files)=}")
    if len(set(files)) != len(files):
        raise RuntimeError("Virtual_source_files contains duplicates, cannot proceed with import.")

    # Properties will raise RuntimeError if internal API has changed
    # If data is not a virtual dataset, they return None
    if files is None or vpath is None:
        raise RuntimeError(f"Expected virtual dataset in {outfile}, but virtual dataset information is not available.")

    logger.info("Writing {} image batches (requested n_jobs: {})...", len(slices), nproc)
    t0 = time.perf_counter()
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        log_parallel_backend(parallel)
        results = parallel(
            delayed(import_batch)(None if reader else iset, reader, sl.start, sl.stop, fn, vpath)
            for sl, fn in zip(slices, files)
        )
    wall = time.perf_counter() - t0
    steps = {k: sum(r[0][k] for r in results) for k in results[0][0]}
    total = sum(steps.values())
    logger.info(
        "Task time {:.1f} s over {:.1f} s wall: {}",
        total, wall, ", ".join(f"{k} {v:.1f} s ({v / total * 100:.0f}%)" for k, v in steps.items() if v > 0),
    )  # fmt: skip
    noverflow = sum(r[1] for r in results)
    if noverflow:
        logger.info("{} pixels above {} (kept exactly)", noverflow, OVERFLOW16 - 1)
    logger.info("Image data writing completed")

    logger.info("Image data import completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_import_data))


if __name__ == "__main__":
    run()
