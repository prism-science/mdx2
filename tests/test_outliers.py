"""Two-sided leave-one-out outlier rejection (ScaledData.mask_outliers, method "two_sided")."""

import numba
import numpy as np
import pytest

from mdx2.scaling import ScaledData


def brute_force(I, sigma, ih, scale, offset, mask, nsigma):
    """One call, written out per group: returns the new mask and (high, low)."""
    mask = mask.copy()
    high = low = 0
    x = I / scale - offset
    w = (scale / sigma) ** 2
    for g in np.unique(ih):
        idx = np.flatnonzero((ih == g) & ~mask)
        if idx.size < 2:
            continue
        r = []
        for i in idx:
            o = idx[idx != i]
            Wr = w[o].sum()
            r.append((x[i] - (w[o] * x[o]).sum() / Wr) / np.sqrt(1 / w[i] + 1 / Wr))
        r = np.array(r)
        j = np.argmax(np.abs(r))
        if abs(r[j]) > nsigma:
            if idx.size == 2:
                mask[idx] = True
                high += 1
                low += 1
            else:
                mask[idx[j]] = True
                high += r[j] > 0
                low += r[j] < 0
    return mask, (high, low)


def data(seed=0, n_groups=400):
    rng = np.random.default_rng(seed)
    mult = rng.integers(1, 9, n_groups)  # includes singletons and pairs
    ih = np.repeat(np.arange(n_groups), mult)
    n = ih.size
    I0 = rng.exponential(100.0, n_groups)
    scale = rng.uniform(0.5, 1.5, n)
    offset = rng.uniform(0, 5, n)
    sigma = np.sqrt(scale * (I0[ih] + offset) + 4.0)
    I = scale * (I0[ih] + offset) + sigma * rng.standard_normal(n)
    spikes = rng.random(n) < 0.05
    I[spikes] += sigma[spikes] * rng.choice([-12.0, 12.0], spikes.sum())
    mask = rng.random(n) < 0.03
    return I, sigma, ih, scale, offset, mask


@pytest.mark.parametrize("nthreads", [1, None])
def test_matches_brute_force(nthreads):
    numba.set_num_threads(nthreads or numba.config.NUMBA_NUM_THREADS)
    I, sigma, ih, scale, offset, mask = data()
    S = ScaledData(I, sigma, ih, scale=scale.copy(), offset=offset.copy(), mask=mask.copy())
    want_mask = mask
    for _ in range(4):  # repeated calls: one more per group each time
        want_mask, (high, low) = brute_force(I, sigma, ih, scale, offset, want_mask, 3.0)
        n = S.mask_outliers(None, 3.0)
        np.testing.assert_array_equal(S.mask, want_mask)
        assert (S.last_outliers["high"], S.last_outliers["low"]) == (high, low)
        assert n == S.last_outliers["total"]


def group(values, sigma=1.0):
    I = np.asarray(values, dtype=np.float64)
    return ScaledData(I, np.full(I.size, sigma), np.zeros(I.size, dtype=int))


def test_single_spike_only_spike_masked():
    for spike, side in ((60.0, "high"), (-40.0, "low")):
        S = group([10.0, 10.5, 9.5, 10.2, 9.8, spike])
        assert S.mask_outliers(None, 5.0) == 1
        assert S.mask.tolist() == [False] * 5 + [True]
        assert S.last_outliers[side] == 1
        assert S.mask_outliers(None, 5.0) == 0  # the rest agree


def test_pairs_and_singletons():
    S = group([10.0, 30.0])  # |r| = 20 / sqrt(2) > 5: no way to tell which one, both go
    assert S.mask_outliers(None, 5.0) == 2 and S.mask.all()
    S = group([10.0, 12.0])  # |r| = 2 / sqrt(2): keep
    assert S.mask_outliers(None, 5.0) == 0
    S = group([1000.0])  # nothing to compare with
    assert S.mask_outliers(None, 5.0) == 0
