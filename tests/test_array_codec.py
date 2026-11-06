"""Large arrays of mdx2 objects are written with ARRAY_COMPRESSION (Blosc LZ4 + byte shuffle; table columns in ~1 MB
chunks) instead of nexusformat's gzip default; small arrays stay uncompressed; data round-trip exactly."""

import h5py
import numpy as np
import pytest

from mdx2.data import HKLTable, Peaks
from mdx2.geometry import Corrections, GridData, MillerIndex
from mdx2.io import TABLE_CHUNK_BYTES, array_field, loadobj, saveobj

BLOSC = 32001


def filters(ds):
    pl = ds.id.get_create_plist()
    return [pl.get_filter(i)[0] for i in range(pl.get_nfilters())]


def datasets(fn):
    out = {}
    with h5py.File(fn, "r") as f:
        f.visititems(
            lambda n, o: out.__setitem__(n, (o.size, o.chunks, filters(o), o.dtype.itemsize))
            if isinstance(o, h5py.Dataset)
            else None
        )
    return out


def check_codec(fn, column=False):
    """Every array above nexusformat's maxsize is Blosc-only (no HDF5 shuffle / gzip), every small one uncompressed."""
    big = 0
    for name, (size, chunks, flt, itemsize) in datasets(fn).items():
        if size > 10000:
            big += 1
            assert flt == [BLOSC], (name, flt)
            if column:  # ~1 MB chunks (the whole column if smaller)
                assert chunks == (min(size, TABLE_CHUNK_BYTES // itemsize),), (name, chunks)
        else:
            assert flt == [], (name, flt)
    assert big > 0


def test_hkl_table(tmp_path):
    rng = np.random.default_rng(0)
    n = 300_000
    intensity = rng.normal(size=n).astype(np.float32)
    intensity[::7] = np.nan
    t = HKLTable(
        rng.random(n).astype(np.float32), rng.random(n), np.zeros(n), ndiv=(1, 2, 3),
        counts=np.arange(n, dtype=np.int32), intensity=intensity,
    )  # fmt: skip
    fn = str(tmp_path / "t.nxs")
    saveobj(t, fn, name="hkl_table")
    check_codec(fn, column=True)
    u = loadobj(fn, "hkl_table")
    assert tuple(u.ndiv) == (1, 2, 3)
    for c in ("h", "k", "l", "counts", "intensity"):
        np.testing.assert_array_equal(getattr(u, c), getattr(t, c))


def test_peaks(tmp_path):
    rng = np.random.default_rng(1)
    p = Peaks(rng.random(50_000), rng.random(50_000), rng.random(50_000))
    fn = str(tmp_path / "p.nxs")
    saveobj(p, fn, name="peaks")
    check_codec(fn, column=True)
    q = loadobj(fn, "peaks")
    for c in ("phi", "iy", "ix"):
        np.testing.assert_array_equal(getattr(q, c), getattr(p, c))


@pytest.mark.parametrize("kind", ["miller_index", "corrections", "grid_data"])
def test_grids_keep_their_chunking(tmp_path, kind):
    rng = np.random.default_rng(2)
    shape = (30, 40, 50)
    if kind == "miller_index":
        obj = MillerIndex(np.arange(30.0), np.arange(40.0), np.arange(50.0), *(rng.random(shape) for _ in range(3)))
    elif kind == "corrections":
        names = ("solid_angle", "efficiency", "attenuation", "polarization", "inverse_lorentz", "d3s")
        maps = {k: rng.random((120, 100)) for k in names}
        obj = Corrections(np.arange(120.0), np.arange(100.0), **maps)
    else:
        obj = GridData(
            (np.arange(30.0), np.arange(40.0), np.arange(50.0)), rng.random(shape), axes_names=["phi", "iy", "ix"]
        )
    fn = str(tmp_path / "g.nxs")
    saveobj(obj, fn, name=kind)
    check_codec(fn)
    back = loadobj(fn, kind)
    for a, b in zip(GridData.data_unstacked.fget(back), GridData.data_unstacked.fget(obj)):
        np.testing.assert_array_equal(a, b)


def test_array_field():
    assert array_field(np.zeros(10), "a").compression is None  # small: as nexusformat, uncompressed
    assert array_field(np.array(["x"] * 20000), "s").compression == "gzip"  # non-numeric: nexusformat's default
    f = array_field(np.zeros(20000, np.float32), "b", column=True)
    assert f.compression is not None and f.compression != "gzip" and not f.shuffle
    assert f.chunks == (20000,)  # whole column when it is under ~1 MB
