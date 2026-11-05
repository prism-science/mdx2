"""Datastore compression (mdx2.import_data): every codec writes and reads back the same image data."""

import h5py
import numpy as np
import pytest
from nexusformat.nexus import NXfield

from mdx2.data import DATASTORE_COMPRESSION, ImageSeries
from mdx2.io import nxload

FILTER = {"blosc-lz4": "32001", "blosc-zstd": "32001", "gzip": "gzip"}  # 32001: Blosc's registered HDF5 filter id


@pytest.mark.parametrize("compression", sorted(DATASTORE_COMPRESSION))
def test_datastore_round_trip(tmp_path, compression):
    rng = np.random.default_rng(0)
    shape, chunks = (12, 40, 50), (4, 20, 25)
    images = rng.poisson(3.0, shape).astype(np.int32)
    images[rng.random(shape) < 0.01] = -1  # masked pixels
    images[0, 0, :5] = [70000, 2**20, 0, 1, 123456]  # counts beyond int16
    phi, iy, ix = np.arange(shape[0]) * 0.1, np.arange(shape[1]), np.arange(shape[2])
    data = NXfield(shape=shape, name="data", chunks=chunks, dtype=np.int32, **DATASTORE_COMPRESSION[compression])
    ims = ImageSeries(phi, iy, ix, data, np.full(shape[0], 0.1))

    # as mdx2.import_data: virtual dataset with one source file per stack of frames, then write the stacks
    out = str(tmp_path / "data.nxs")
    ims.save(out, virtual=True, source_directory=str(tmp_path / "datastore"))
    loaded = ImageSeries.load(out)
    for sl, fn in zip(loaded.chunk_slice_along_axis(0), loaded.virtual_source_files):
        nxload(fn, "r+")[loaded.virtual_dataset_path][:, :, :] = images[sl]

    loaded = ImageSeries.load(out)
    np.testing.assert_array_equal(np.asarray(loaded.data.nxdata), images)
    for sl in loaded.chunk_slice_iterator():  # chunk reads, as integrate / reintegrate
        np.testing.assert_array_equal(loaded[sl].data, images[sl])
    with h5py.File(loaded.virtual_source_files[0], "r") as f:
        ds = f[loaded.virtual_dataset_path]
        assert FILTER[compression] in ds._filters, list(ds._filters)
        assert ds.chunks == chunks


def test_unknown_compression():
    with pytest.raises(ValueError, match="unknown compression"):
        ImageSeries.from_expt("does-not-matter.expt", compression="lzma")
