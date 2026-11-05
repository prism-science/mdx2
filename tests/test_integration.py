"""Fast integration engine (mdx2.integration.bin_chunk, chunk-column tasks) vs the reference code paths."""

import numpy as np
import pandas as pd
import pytest

from mdx2.command_line.integrate import Parameters as IntParams
from mdx2.command_line.integrate import run_integrate
from mdx2.command_line.reintegrate import Parameters as ReintParams
from mdx2.command_line.reintegrate import run_reintegrate
from mdx2.geometry import load_mask
from mdx2.integration import bin_chunk, chunk_columns
from mdx2.io import loadobj

from .test_peak_mask import make_masks, write_sweep


@pytest.fixture
def sweep(tmp_path):
    return write_sweep(tmp_path)


def masks(sweep):
    tmp_path, geom, dat, pk = sweep
    fs, fd = make_masks(tmp_path, geom, dat, pk)
    _, fb = make_masks(tmp_path, geom, dat, pk, bragg=True)
    return {"none": None, "static": load_mask(fs), "dynamic": load_mask(fd), "bragg": load_mask(fb)}


def test_bin_chunk_matches_index_bin(sweep):
    tmp_path, geom, dat, pk = sweep
    mi, ims = loadobj(geom, "miller_index"), loadobj(dat, "image_series")
    assert len(chunk_columns(ims)) > 1
    for name, mask in masks(sweep).items():
        for ndiv in ((1, 1, 1), (2, 3, 2)):
            for sl in ims.chunk_slice_iterator():
                ref = ims[sl].index(mi, mask=None if mask is None else mask.for_chunk(sl))
                ref.ndiv = ndiv
                ref = ref.bin(count_name="pixels")
                new = bin_chunk(ims, sl, mi, mask, ndiv)
                if len(ref) == 0:
                    assert new is None
                    continue
                assert list(new._data_keys) == list(ref._data_keys)
                for c in ["h", "k", "l"] + list(ref._data_keys):
                    a, b = getattr(ref, c), getattr(new, c)
                    assert a.dtype == b.dtype, (name, c)
                    np.testing.assert_array_equal(a, b, err_msg=f"{name} {ndiv} {c}")


@pytest.mark.parametrize("output", ["counts", "intensity"])
def test_reintegrate_engines(sweep, output):
    tmp_path, geom, dat, pk = sweep
    _, fd = make_masks(tmp_path, geom, dat, pk)
    tabs = {}
    for engine in ("reference", "fast"):
        out = str(tmp_path / f"reint_{engine}.nxs")
        run_reintegrate(ReintParams(geom=geom, data=dat, mask=fd, subdivide=(2, 2, 2), output=output, outfile=out,
                                    engine=engine))  # fmt: skip
        tabs[engine] = loadobj(out, "hkl_table")
    a, b = tabs["reference"], tabs["fast"]
    assert len(a) == len(b) > 0
    for c in ["h", "k", "l"] + sorted(a._data_keys):
        x, y = getattr(a, c), getattr(b, c)
        if c in ("h", "k", "l", "counts"):
            np.testing.assert_array_equal(x, y, err_msg=c)
        else:  # float32 sums accumulated in a different order
            np.testing.assert_allclose(x, y, rtol=2e-6, equal_nan=True, err_msg=c)


def integrate(sweep, engine, name):
    tmp_path, geom, dat, pk = sweep
    _, fd = make_masks(tmp_path, geom, dat, pk)
    out = str(tmp_path / f"int_{engine}_{name}.nxs")
    run_integrate(IntParams(geom=geom, data=dat, mask=fd, subdivide=(2, 2, 2), max_spread=1.0, outfile=out,
                            engine=engine))  # fmt: skip
    t = loadobj(out, "hkl_table")
    return pd.DataFrame({c: getattr(t, c) for c in ["h", "k", "l"] + list(t._data_keys)})


def test_integrate_single_column_identical(tmp_path):
    """With one chunk column (no module boundaries), the fast engine reproduces the reference exactly."""
    sweep = write_sweep(tmp_path, chunks=(4, 60, 70))
    assert len(chunk_columns(loadobj(sweep[2], "image_series"))) == 1
    a, b = integrate(sweep, "reference", "1col"), integrate(sweep, "fast", "1col")
    assert list(a.columns) == list(b.columns) and len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_integrate_per_module(sweep):
    """Several chunk columns: observations are kept per module, so there can be more of them; totals conserved."""
    a, b = integrate(sweep, "reference", "multi"), integrate(sweep, "fast", "multi")
    assert len(b) >= len(a) > 0
    for c in ("counts", "pixels"):
        assert a[c].sum() == b[c].sum(), c
    np.testing.assert_allclose(a.seconds.sum(), b.seconds.sum(), rtol=1e-6)
    # observations of a voxel never straddle a module boundary: per (h, k, l, n), same totals or a split
    ga = a.groupby(["h", "k", "l"]).counts.sum()
    gb = b.groupby(["h", "k", "l"]).counts.sum()
    pd.testing.assert_series_equal(ga, gb)
