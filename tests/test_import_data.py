"""mdx2.import_data batch assembly: the uint16 buffer + overflow list writes back the exact int32 frames."""

import h5py
import numpy as np
import pytest

from mdx2._import_kernels import MASKED16, OVERFLOW16
from mdx2.command_line.import_data import import_batch, pack_frame, write_chunks
from mdx2.data import DATASTORE_COMPRESSION
from mdx2.dxtbx_machinery import NXmxFrameReader


def frames(rng, n=5, ny=37, nx=53):
    f = rng.poisson(20.0, (n, ny, nx)).astype(np.int32)
    f[rng.random(f.shape) < 0.05] = -1  # masked
    f[0, 0, :6] = [OVERFLOW16 - 1, OVERFLOW16, MASKED16, 70000, 2**31 - 1, 0]  # around and above the uint16 codes
    f[1, 3, 4] = -2  # a negative value other than the mask value (kept exactly)
    f[2, -1, -1] = 128080
    f[4] = -1  # a fully masked frame
    return f


@pytest.mark.parametrize("compression", ["blosc-lz4", "gzip"])
@pytest.mark.parametrize("chunks", [(5, 37, 53), (5, 10, 20)])  # one chunk; edge chunks of partial size
def test_pack_and_write_round_trip(tmp_path, compression, chunks):
    rng = np.random.default_rng(1)
    data = frames(rng)
    buf = np.empty(data.shape, dtype=np.uint16)
    overflow = []
    for i in range(data.shape[0]):
        pack_frame(buf[i], data[i], i, overflow)
    assert sum(o[0].size for o in overflow) == 6  # 65534, 65535, 70000, 2**31-1, -2, 128080
    fn = tmp_path / "batch.h5"
    with h5py.File(fn, "w") as f:
        ds = f.create_dataset(
            "d", shape=data.shape, dtype=np.int32, chunks=chunks, **DATASTORE_COMPRESSION[compression]
        )
        write_chunks(ds, buf, overflow, ds.chunks)
    with h5py.File(fn, "r") as f:
        np.testing.assert_array_equal(f["d"][()], data)


def test_pack_without_overflow():
    frame = np.arange(12, dtype=np.int32).reshape(3, 4)
    frame[0, 0] = -1
    buf = np.empty((3, 4), dtype=np.uint16)
    overflow = []
    pack_frame(buf, frame, 0, overflow)
    assert overflow == []
    assert buf[0, 0] == MASKED16
    np.testing.assert_array_equal(buf.ravel()[1:], np.arange(1, 12))


class FakeImageSet:
    """Stands in for mdx2.dxtbx_machinery.ImageSet: frames from an array."""

    def __init__(self, frames):
        self.frames = frames
        self.frame_shape = frames.shape[1:]

    def read_frame(self, ind):
        return self.frames[ind].copy()


@pytest.mark.parametrize("bright", [False, True])  # True: >1% of pixels above 65533, batch switches to int32
def test_import_batch(tmp_path, bright):
    rng = np.random.default_rng(2)
    data = frames(rng, n=6)
    if bright:
        data[3:] = rng.integers(60000, 200000, data[3:].shape, dtype=np.int32)
    fn = str(tmp_path / "batch.h5")
    with h5py.File(fn, "w") as f:
        f.create_dataset(
            "d", shape=data.shape, dtype=np.int32, chunks=(6, 10, 20), **DATASTORE_COMPRESSION["blosc-lz4"]
        )
    T, noverflow = import_batch(FakeImageSet(data), None, 0, data.shape[0], fn, "d")
    assert noverflow == int(np.sum(data > 65533)) + (0 if bright else 1)  # (the -2 pixel is an overflow entry too)
    with h5py.File(fn, "r") as f:
        np.testing.assert_array_equal(f["d"][()], data)


def write_nxmx_like(tmp_path, frames_per_file=(4, 3), ny=6, nx=5):
    """A master file with external links entry/data/data_NNNNNN to data files, as written by Eiger file writers."""
    rng = np.random.default_rng(3)
    master = tmp_path / "master.h5"
    allframes = []
    with h5py.File(master, "w") as m:
        g = m.create_group("entry/data")
        for k, n in enumerate(frames_per_file, start=1):
            fn = tmp_path / f"data_{k:06d}.h5"
            d = rng.integers(0, 70000, (n, ny, nx)).astype(np.uint32)
            d[:, 0, 0] = 2**32 - 1  # gap / dead pixel code
            with h5py.File(fn, "w") as f:
                f.create_dataset("entry/data/data", data=d, chunks=(1, ny, nx))
            g[f"data_{k:06d}"] = h5py.ExternalLink(fn.name, "entry/data/data")
            allframes.append(d)
    return str(master), np.concatenate(allframes)


@pytest.mark.parametrize("held_swmr", [None, True, False])  # files already open in this process (as by dxtbx)
def test_nxmx_reader_any_swmr_flag(tmp_path, held_swmr):
    master, data = write_nxmx_like(tmp_path)
    held = None
    if held_swmr is not None:
        held = h5py.File(master, "r", swmr=held_swmr)
        held["entry/data/data_000002"][0]  # opens the linked data file too
    lo, hi = 0.0, 65000.0
    r = NXmxFrameReader(master, ["data_000001", "data_000002"], [4, 3], range(7), data.shape[1:], (lo, hi), None)
    for ind in range(7):
        v = data[ind].astype(np.int64)
        expected = np.where((v >= lo) & (v <= hi), v, -1).astype(np.int32)
        np.testing.assert_array_equal(r.read_frame(ind), expected)
    r.close()
    if held is not None:
        held.close()
