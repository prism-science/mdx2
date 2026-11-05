"""mdx2.import_data: uint16 batch assembly (exact), threaded decode / compress / write, and the NXmx frame reader."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import bitshuffle
import dxtbx.model.experiment_list  # noqa: F401  (loads cctbx's omptbx_ext; see test_bitshuffle_loaded_...)
import h5py
import hdf5plugin
import numpy as np
import pytest
from nexusformat.nexus import NXfield

import mdx2
from mdx2._import_kernels import MASKED16, OVERFLOW16, pack_frame
from mdx2.command_line.import_data import direct_write_codec, import_batch, write_chunks
from mdx2.data import DATASTORE_COMPRESSION, ImageSeries
from mdx2.dxtbx_machinery import NXmxFrameReader, is_bitshuffle_lz4_frames


def frames(rng, n=5, ny=37, nx=53):
    f = rng.poisson(20.0, (n, ny, nx)).astype(np.int32)
    f[rng.random(f.shape) < 0.05] = -1  # masked
    f[0, 0, :6] = [OVERFLOW16 - 1, OVERFLOW16, MASKED16, 70000, 2**31 - 1, 0]  # around and above the uint16 codes
    f[1, 3, 4] = -2  # a negative value other than the mask value (kept exactly)
    f[2, -1, -1] = 128080
    f[4] = -1  # a fully masked frame
    return f


def pack_all(data):
    buf = np.empty(data.shape, dtype=np.uint16)
    parts = []
    for i in range(data.shape[0]):
        ov = pack_frame(data[i], buf[i])
        if ov is not None:
            parts.append((np.full(ov[0].size, i, dtype=np.int64), *ov))
    return buf, (tuple(np.concatenate(a) for a in zip(*parts)) if parts else None)


@pytest.mark.parametrize("compression", ["blosc-lz4", "blosc-zstd", "gzip", "blosc-lz4+shuffle"])
@pytest.mark.parametrize("chunks", [(5, 37, 53), (5, 10, 20)])  # one chunk; edge chunks of partial size
@pytest.mark.parametrize("threaded", [False, True])  # True: numcodecs Blosc in threads + write_direct_chunk
def test_pack_and_write_round_trip(tmp_path, compression, chunks, threaded):
    data = frames(np.random.default_rng(1))
    buf, ov = pack_all(data)
    assert ov[0].size == 6  # 65534, 65535, 70000, 2**31-1, -2, 128080
    # blosc-lz4+shuffle: HDF5 byte-shuffle before Blosc (datastores written before shuffle=False): h5py writes
    opts = dict(DATASTORE_COMPRESSION[compression.split("+")[0]])
    if compression.endswith("+shuffle"):
        opts["shuffle"] = True
    fn = tmp_path / "batch.h5"
    with h5py.File(fn, "w") as f:
        ds = f.create_dataset("d", shape=data.shape, dtype=np.int32, chunks=chunks, **opts)
        codec = direct_write_codec(ds, compression) if threaded else None
        assert threaded is False or (codec is not None) == (compression in ("blosc-lz4", "blosc-zstd"))
        with ThreadPoolExecutor(3) as pool:
            times = write_chunks(ds, buf, ov, codec, pool if threaded else None)
    assert ("compress" in times) == (codec is not None)
    with h5py.File(fn, "r") as f:  # read back through HDF5's own filter
        np.testing.assert_array_equal(f["d"][()], data)


def test_pack_without_overflow():
    frame = np.arange(12, dtype=np.int32).reshape(3, 4)
    frame[0, 0] = -1
    buf = np.empty((3, 4), dtype=np.uint16)
    assert pack_frame(frame, buf) is None
    assert buf[0, 0] == MASKED16
    np.testing.assert_array_equal(buf.ravel()[1:], np.arange(1, 12))


class FakeSource:
    """A frame source (see import_batch) from an array; records which threads decoded frames."""

    def __init__(self, frames, concurrent):
        self.frames = frames
        self.frame_shape = frames.shape[1:]
        self.concurrent = concurrent
        self.threads = set()

    def read_frame(self, ind):
        return self.frames[ind].copy()

    def pack(self, ind, buf):
        self.threads.add(threading.get_ident())
        return pack_frame(self.frames[ind], buf), {"read": 0.0, "pack": 0.0}


@pytest.mark.parametrize("bright", [False, True])  # True: >1% of pixels above 65533, batch switches to int32
@pytest.mark.parametrize("nthreads,concurrent", [(1, True), (3, True), (3, False)])
@pytest.mark.parametrize("compression", ["blosc-lz4", "gzip"])
def test_import_batch(tmp_path, bright, nthreads, concurrent, compression):
    rng = np.random.default_rng(2)
    data = frames(rng, n=9)
    if bright:
        data[3:] = rng.integers(60000, 200000, data[3:].shape, dtype=np.int32)
    fn = str(tmp_path / "batch.h5")
    with h5py.File(fn, "w") as f:
        f.create_dataset(
            "d", shape=data.shape, dtype=np.int32, chunks=(9, 10, 20), **DATASTORE_COMPRESSION[compression]
        )
    source = FakeSource(data, concurrent)
    T, W, noverflow = import_batch(source, 0, data.shape[0], fn, "d", compression, nthreads)
    assert noverflow == int(np.sum(data > 65533)) + (0 if bright else 1)  # (the -2 pixel is an overflow entry too)
    assert len(source.threads) == 1 or (concurrent and nthreads > 1)
    with h5py.File(fn, "r") as f:
        np.testing.assert_array_equal(f["d"][()], data)


def write_nxmx_like(tmp_path, frames_per_file=(4, 3), ny=6, nx=5, bitshuffle=False):
    """A master file with external links entry/data/data_NNNNNN to data files, as written by Eiger file writers
    (bitshuffle: one bitshuffle-LZ4 chunk per frame, as DECTRIS)."""
    rng = np.random.default_rng(3)
    master = tmp_path / "master.h5"
    allframes = []
    opts = dict(compression=hdf5plugin.Bitshuffle(cname="lz4")) if bitshuffle else {}
    with h5py.File(master, "w") as m:
        g = m.create_group("entry/data")
        for k, n in enumerate(frames_per_file, start=1):
            fn = tmp_path / f"data_{k:06d}.h5"
            d = rng.integers(0, 70000, (n, ny, nx)).astype(np.uint32)
            d[:, 0, 0] = 2**32 - 1  # gap / dead pixel code
            with h5py.File(fn, "w") as f:
                f.create_dataset("entry/data/data", data=d, chunks=(1, ny, nx), **opts)
            g[f"data_{k:06d}"] = h5py.ExternalLink(fn.name, "entry/data/data")
            allframes.append(d)
    return str(master), np.concatenate(allframes)


def expected_frames(data, lo, hi):
    v = data.astype(np.int64)
    return np.where((v >= lo) & (v <= hi), v, -1).astype(np.int32)


@pytest.mark.parametrize("bitshuffle", [False, True])
@pytest.mark.parametrize("held_swmr", [None, True, False])  # files already open in this process (as by dxtbx)
def test_nxmx_reader(tmp_path, bitshuffle, held_swmr):
    master, data = write_nxmx_like(tmp_path, bitshuffle=bitshuffle)
    with h5py.File(master, "r") as f:
        assert is_bitshuffle_lz4_frames(f["entry/data/data_000001"], data.shape[1:]) == bitshuffle
    held = None
    if held_swmr is not None:
        held = h5py.File(master, "r", swmr=held_swmr)
        held["entry/data/data_000002"][0]  # opens the linked data file too
    lo, hi = 0.0, 65000.0
    r = NXmxFrameReader(
        master, ["data_000001", "data_000002"], [4, 3], range(7), data.shape[1:], (lo, hi), None, direct=bitshuffle
    )
    assert r.direct == bitshuffle
    expected = expected_frames(data, lo, hi)
    for ind in range(7):  # across the data-file boundary (3 | 4)
        np.testing.assert_array_equal(r.read_frame(ind), expected[ind])
    # concurrent pack() calls (as import_batch with threads)
    r.open()
    buf = np.empty(data.shape, dtype=np.uint16)
    with ThreadPoolExecutor(4) as pool:
        overflows = list(pool.map(lambda i: r.pack(i, buf[i])[0], range(7)))
    assert all(ov is None for ov in overflows)  # values <= hi = 65000 < 65534
    np.testing.assert_array_equal(np.where(buf == MASKED16, -1, buf.astype(np.int32)), expected)
    r.close()
    if held is not None:
        held.close()


@pytest.mark.parametrize("compression", ["blosc-lz4", "blosc-zstd", "gzip"])
def test_import_batch_into_datastore(tmp_path, compression):
    """As mdx2.import_data: datastore files made by ImageSeries.save (nexusformat), then filled by import_batch."""
    data = frames(np.random.default_rng(4), n=12, ny=40, nx=50)
    shape, chunks = data.shape, (4, 20, 25)
    field = NXfield(shape=shape, name="data", chunks=chunks, dtype=np.int32, **DATASTORE_COMPRESSION[compression])
    ims = ImageSeries(
        np.arange(shape[0]) * 0.1, np.arange(shape[1]), np.arange(shape[2]), field, np.full(shape[0], 0.1)
    )
    out = str(tmp_path / "data.nxs")
    ims.save(out, virtual=True, source_directory=str(tmp_path / "datastore"))
    loaded = ImageSeries.load(out)
    source = FakeSource(data, concurrent=True)
    for sl, fn in zip(loaded.chunk_slice_along_axis(0), loaded.virtual_source_files):
        T, W, _ = import_batch(source, sl.start, sl.stop, fn, loaded.virtual_dataset_path, compression, nthreads=3)
        assert W["direct write"] == (compression != "gzip")
    with h5py.File(loaded.virtual_source_files[0], "r") as f:
        plist = f[loaded.virtual_dataset_path].id.get_create_plist()
        codes = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
    assert codes == ([2, 1] if compression == "gzip" else [32001])  # gzip: shuffle (2), then deflate (1)
    np.testing.assert_array_equal(np.asarray(ImageSeries.load(out).data.nxdata), data)


def test_bitshuffle_loaded_before_cctbx_openmp_stubs():
    """mdx2 loads bitshuffle (OpenMP) before cctbx's omptbx stubs; with them loaded too, decompression works (an
    extension loaded after omptbx binds to its stub omp_* functions and crashes in bitshuffle's parallel regions)."""
    assert mdx2.BITSHUFFLE_SAFE
    x = (np.arange(600 * 500, dtype=np.uint32) % 997).reshape(600, 500)
    y = bitshuffle.decompress_lz4(bitshuffle.compress_lz4(x), x.shape, x.dtype)
    np.testing.assert_array_equal(x, y)
    assert "omptbx_ext" in sys.modules
