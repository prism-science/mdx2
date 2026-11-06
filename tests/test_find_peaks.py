"""mdx2.find_peaks: the fast engine (chunk-column tasks, compiled threshold) gives the same peaks, in the same order,
and the same peak model as the reference engine; MillerIndex.interpolate is unchanged by computing per component."""

import numpy as np
import pytest

from mdx2._integration_kernels import threshold_indices_kernel
from mdx2.command_line.find_peaks import Parameters, run_find_peaks
from mdx2.geometry import GridData
from mdx2.io import loadobj

from .test_peak_mask import write_sweep


@pytest.fixture
def sweep(tmp_path):
    return write_sweep(tmp_path, chunks=(4, 20, 25))  # several chunk columns


def run(sweep, engine, threshold, nproc):
    tmp_path, geom, dat, _ = sweep
    out = str(tmp_path / f"peaks_{engine}_{threshold}_{nproc}.nxs")
    run_find_peaks(Parameters(geom=geom, data=dat, count_threshold=threshold, outfile=out, engine=engine, nproc=nproc))
    return {name: loadobj(out, name) for name in ("peaks", "outliers", "peak_model")}


@pytest.mark.parametrize("threshold", [20.0, -5.0])  # -5: every pixel except the masked ones (value -1)
@pytest.mark.parametrize("nproc", [1, 2])
def test_engines_identical(sweep, threshold, nproc):
    ref = run(sweep, "reference", threshold, 1)
    new = run(sweep, "fast", threshold, nproc)
    for name in ("peaks", "outliers"):
        for ax in ("phi", "iy", "ix"):
            np.testing.assert_array_equal(getattr(new[name], ax), getattr(ref[name], ax), err_msg=f"{name}.{ax}")
    np.testing.assert_array_equal(new["peak_model"].r0, ref["peak_model"].r0)
    np.testing.assert_array_equal(new["peak_model"].sigma, ref["peak_model"].sigma)
    if threshold < 0:
        data = np.asarray(loadobj(sweep[2], "image_series").data)
        assert ref["peaks"].size == int(np.sum(data != -1))


def test_miller_index_interpolate_per_component(sweep):
    """MillerIndex.interpolate (per component) == the previous path through GridData (stacked h, k, l)."""
    _, geom, dat, _ = sweep
    mi = loadobj(geom, "miller_index")
    ims = loadobj(dat, "image_series")
    rng = np.random.default_rng(0)
    n = 500
    phi = rng.uniform(ims.phi[0], ims.phi[-1], n)
    iy = rng.uniform(ims.iy[0], ims.iy[-1], n)
    ix = rng.uniform(ims.ix[0], ims.ix[-1], n)
    old = GridData.interpolate(mi, phi, iy, ix)
    for i, a in enumerate(mi.interpolate(phi, iy, ix)):
        np.testing.assert_array_equal(a, old[..., i])


def test_threshold_kernel():
    rng = np.random.default_rng(1)
    data = rng.poisson(5.0, (3, 11, 13)).astype(np.int32)
    data[rng.random(data.shape) < 0.1] = -1
    for threshold in (-5.0, 0.0, 6.5, 100.0):
        expected = np.flatnonzero((data != -1) & (data > threshold))
        np.testing.assert_array_equal(threshold_indices_kernel(data, threshold, -1), expected)
        assert np.array_equal(
            np.argwhere(np.ma.masked_equal(data, -1) > threshold), np.argwhere((data != -1) & (data > threshold))
        )
