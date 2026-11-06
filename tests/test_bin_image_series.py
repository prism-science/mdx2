"""mdx2.bin_image_series: the vectorized per-bin means are identical to the previous masked-array implementation."""

import numpy as np
import pytest

from mdx2.command_line.bin_image_series import Parameters, bin_slab, run_bin_image_series
from mdx2.io import loadobj
from mdx2.utils import slice_sections

from .test_peak_mask import write_sweep


def reference(data, maskval, valid_range, mask, sl_1, sl_2):
    """mdx2 <= 1.0.5 binslab: masked arrays and a loop over bins."""
    out = np.empty([len(sl_1), len(sl_2)], dtype=float)
    tmp = np.ma.masked_equal(data, maskval, copy=False)
    if valid_range is not None:
        tmp = np.ma.masked_outside(tmp, valid_range[0], valid_range[1], copy=False)
    if mask is not None:
        tmp = np.ma.masked_where(mask, tmp, copy=False)
    for iy, sy in enumerate(sl_1):
        for ix, sx in enumerate(sl_2):
            val = tmp[:, sy, sx].mean()
            out[iy, ix] = np.nan if np.ma.is_masked(val) else val
    return out


@pytest.mark.parametrize("valid_range", [None, (0, 100)])
@pytest.mark.parametrize("with_mask", [False, True])
def test_bin_slab_matches_masked_arrays(valid_range, with_mask):
    rng = np.random.default_rng(0)
    data = rng.poisson(20.0, (5, 47, 61)).astype(np.int32)
    data[rng.random(data.shape) < 0.05] = -1  # detector mask value
    data[rng.random(data.shape) < 0.01] = 5000  # outside valid_range
    mask = rng.random(data.shape) < 0.1 if with_mask else None
    sl_1, sl_2 = slice_sections(47, 5), slice_sections(61, 4)  # bins of uneven size
    data[:, sl_1[0], sl_2[0]] = -1  # a bin with no valid pixels
    y0, x0 = np.array([s.start for s in sl_1]), np.array([s.start for s in sl_2])
    new = bin_slab(data, -1, valid_range, mask, y0, x0)
    ref = reference(data, -1, valid_range, mask, sl_1, sl_2)
    assert np.isnan(new[0, 0]) and np.isnan(ref[0, 0])
    np.testing.assert_array_equal(new, ref)


def test_run_bin_image_series(tmp_path):
    tmp_path, geom, dat, pk = write_sweep(tmp_path, chunks=(4, 20, 25))
    out = str(tmp_path / "binned.nxs")
    run_bin_image_series(Parameters(data=dat, bins=(2, 10, 10), valid_range=(0, 100), outfile=out, nproc=2))
    b = loadobj(out, "binned_image_series")
    ims = loadobj(dat, "image_series")
    data = np.asarray(ims.data)
    sl_0, sl_1, sl_2 = (slice_sections(n, int(np.ceil(n / k))) for n, k in zip(data.shape, (2, 10, 10)))
    times = np.array([ims.exposure_times[s].mean() for s in sl_0])
    ref = np.stack([reference(data[s], -1, (0, 100), None, sl_1, sl_2) for s in sl_0]) / times[:, None, None]
    np.testing.assert_array_equal(b.data, ref)
