"""
Import x-ray image data using the dxtbx machinery
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from numcodecs import Blosc, blosc
from simple_parsing import field

import mdx2
from mdx2._import_kernels import OVERFLOW16, unpack_kernel
from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.data import ImageSeries
from mdx2.dxtbx_machinery import ImageSet

blosc.use_threads = False  # numcodecs: contextual Blosc calls, which release the GIL (and no Blosc-internal threads)


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
    nthreads: int = 1
    """threads per process: decoding frames (NXmx: bitshuffle-LZ4 chunks decompressed outside HDF5) and compressing +
    writing datastore chunks (Blosc codecs). Memory grows with processes (one batch buffer each), not with threads."""
    reader: str = field(default="auto", choices=["auto", "dxtbx"])
    """auto: read NXmx (e.g. Eiger) files directly with h5py and apply the dxtbx mask in numpy, ~2.5x faster (the first
    frame is checked against dxtbx; if it differs, dxtbx is used); other formats are read with dxtbx.
    dxtbx: always read with dxtbx."""


# ---- one batch of frames (a datastore file), in a worker -------------------------------------------------------------


# numcodecs Blosc with the settings of DATASTORE_COMPRESSION (hdf5plugin's Blosc filter, 32001, reads the same frames):
# datastore chunks are compressed in threads and written with write_direct_chunk, outside HDF5's filters and lock
DIRECT_WRITE_CODECS = {
    "blosc-lz4": dict(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE),
    "blosc-zstd": dict(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE),
}
BLOSC_FILTER = 32001
BLOSC_COMPRESSORS = {"blosclz": 0, "lz4": 1, "lz4hc": 2, "snappy": 3, "zlib": 4, "zstd": 5}


def direct_write_codec(dataset, compression):
    """The numcodecs codec for writing raw chunks of dataset, or None (then write with h5py): only if the dataset's
    filter pipeline is exactly hdf5plugin's Blosc filter with the settings of DIRECT_WRITE_CODECS[compression]
    (cd_values: ..., clevel, shuffle, compressor)."""
    opts = DIRECT_WRITE_CODECS.get(compression)
    if opts is None:
        return None
    plist = dataset.id.get_create_plist()
    if plist.get_nfilters() != 1:
        return None
    code, _, values, _ = plist.get_filter(0)
    if code != BLOSC_FILTER or len(values) < 7:
        return None
    if tuple(values[4:7]) != (opts["clevel"], opts["shuffle"], BLOSC_COMPRESSORS[opts["cname"]]):
        return None
    return Blosc(**opts)


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


def write_chunks(dataset, buf, overflow_arrays, codec=None, pool=None):
    """Write the uint16 assembly buffer to an int32 HDF5 dataset one chunk column at a time (restoring -1 and the
    overflow values), so that no full-size int32 copy is made.

    codec (a numcodecs codec matching the dataset's filter) and pool (a thread pool): each chunk is unpacked and
    compressed in a thread and written with write_direct_chunk. Otherwise each chunk column is written with h5py.
    Returns step times summed over chunks."""
    n, ny, nx = buf.shape
    cz, cy, cx = dataset.chunks
    tiles = [(y0, x0) for y0 in range(0, ny, cy) for x0 in range(0, nx, cx)]

    def one(tile):
        y0, x0 = tile
        t0 = time.perf_counter()
        out = unpack(buf[:, y0 : y0 + cy, x0 : x0 + cx], overflow_arrays, 0, y0, x0)
        t1 = time.perf_counter()
        if codec is None:
            dataset[:, y0 : y0 + out.shape[1], x0 : x0 + out.shape[2]] = out
            return {"unpack": t1 - t0, "write": time.perf_counter() - t1}
        if out.shape != (cz, cy, cx):  # edge chunk (or a short last batch): HDF5 stores full-size chunks
            full = np.zeros((cz, cy, cx), dtype=np.int32)
            full[: out.shape[0], : out.shape[1], : out.shape[2]] = out
            out = full
        data = codec.encode(out)
        t2 = time.perf_counter()
        dataset.id.write_direct_chunk((0, y0, x0), data)
        return {"unpack": t1 - t0, "compress": t2 - t1, "write": time.perf_counter() - t2}

    times = {}
    for t in pool.map(one, tiles) if (pool is not None and codec is not None) else map(one, tiles):
        for k, v in t.items():
            times[k] = times.get(k, 0.0) + v
    return times


MAX_OVERFLOW_FRACTION = 0.01  # beyond this, a batch is assembled as int32 instead (the overflow list would be larger)


def import_batch(source, istart, istop, filename, datapath, compression=None, nthreads=1):
    """Read frames istart..istop-1 into a uint16 buffer (+ overflow list) and write them to the datastore file.

    source: a frame source (ImageSet: dxtbx, one thread; NXmxFrameReader: thread-safe), with frame_shape, concurrent,
    pack(ind, buf) -> (overflow (y, x, values) or None, step times) and read_frame(ind). nthreads: threads in this
    process, for decoding (if source.concurrent) and for compressing + writing chunks (Blosc codecs).

    Memory: 2 bytes per pixel for the batch, plus per thread one decoded frame and one int32 + compressed chunk. If more
    than MAX_OVERFLOW_FRACTION of the pixels exceed 65533, the batch is assembled as int32 instead (4 bytes per pixel).
    Returns (step times summed over threads, phase wall times, number of overflow pixels)."""
    n = istop - istart
    ny, nx = source.frame_shape
    T, W = {}, {"decode": 0.0, "write": 0.0}

    def add(times):
        for k, v in times.items():
            T[k] = T.get(k, 0.0) + v

    if hasattr(source, "open"):
        source.open()
    pool = ThreadPoolExecutor(nthreads) if nthreads > 1 else None
    decode_pool = pool if source.concurrent else None
    buf = np.empty((n, ny, nx), dtype=np.uint16)
    big = None  # int32 batch, if the uint16 buffer was abandoned
    overflow, noverflow = [], 0

    def decode(i):
        return i, *source.pack(istart + i, buf[i])

    t0 = time.perf_counter()
    group = 2 * nthreads if decode_pool is not None else 1  # check the overflow budget after each group
    i0 = 0
    while i0 < n:
        idx = range(i0, min(i0 + group, n))
        for i, ov, times in decode_pool.map(decode, idx) if decode_pool is not None else map(decode, idx):
            add(times)
            if ov is not None:
                overflow.append((np.full(ov[0].size, i, dtype=np.int64), *ov))
                noverflow += ov[0].size
        i0 = idx.stop
        if noverflow > MAX_OVERFLOW_FRACTION * buf.size:  # switch this batch to int32 (rare: very bright data)
            ov = tuple(np.concatenate(a) for a in zip(*overflow))
            big = np.empty((n, ny, nx), dtype=np.int32)
            big[:i0] = unpack(buf[:i0], ov)
            del buf, ov
            overflow = []
            for i in range(i0, n):
                t = time.perf_counter()
                big[i] = source.read_frame(istart + i)
                add({"read": time.perf_counter() - t})
            break
    t1 = time.perf_counter()
    W["decode"] = t1 - t0
    with h5py.File(filename, "r+") as fh:
        dataset = fh[datapath]
        if big is None:
            ov = tuple(np.concatenate(a) for a in zip(*overflow)) if overflow else None
            codec = direct_write_codec(dataset, compression)
            W["direct write"] = codec is not None
            add(write_chunks(dataset, buf, ov, codec, pool))
        else:
            t = time.perf_counter()
            dataset[:, :, :] = big
            add({"write": time.perf_counter() - t})
    W["write"] = time.perf_counter() - t1
    if pool is not None:
        pool.shutdown()
    return T, W, (int(np.sum(big > OVERFLOW16 - 1)) if big is not None else noverflow)


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
                if reader.direct:
                    how = "raw bitshuffle-LZ4 chunks, decompressed outside HDF5"
                elif not mdx2.BITSHUFFLE_SAFE:
                    how = "h5py; bitshuffle not used: cctbx was loaded before mdx2 (OpenMP stubs)"
                else:
                    how = "h5py"
                logger.info(
                    "Reading frames directly from the NXmx data files ({}; first frame identical to dxtbx)", how
                )
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

    source = reader if reader is not None else iset
    if params.nthreads > 1 and not source.concurrent:
        logger.info(
            "Frames are decoded in one thread per process (dxtbx); {} threads compress and write", params.nthreads
        )
    logger.info(
        "Writing {} image batches (requested n_jobs: {}, threads per job: {})...", len(slices), nproc, params.nthreads
    )
    t0 = time.perf_counter()
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        log_parallel_backend(parallel)
        results = parallel(
            delayed(import_batch)(source, sl.start, sl.stop, fn, vpath, params.compression, params.nthreads)
            for sl, fn in zip(slices, files)
        )
    wall = time.perf_counter() - t0
    steps = {}
    for r in results:
        for k, v in r[0].items():
            steps[k] = steps.get(k, 0.0) + v
    total = sum(steps.values())
    logger.info(
        "Thread time {:.1f} s over {:.1f} s wall: {}",
        total, wall, ", ".join(f"{k} {v:.1f} s ({v / total * 100:.0f}%)" for k, v in steps.items() if v > 0),
    )  # fmt: skip
    logger.info(
        "Batch phases (summed): decode {:.1f} s, write {:.1f} s; raw Blosc chunks written in {} of {} batches",
        sum(r[1]["decode"] for r in results), sum(r[1]["write"] for r in results),
        sum(bool(r[1].get("direct write")) for r in results), len(results),
    )  # fmt: skip
    noverflow = sum(r[2] for r in results)
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
