"""mdx2.scaling (compiled, multi-threaded) must reproduce mdx2.scaling_reference (same algorithm) on synthetic data.

Quick (seconds); run it on a new machine or numba version before a large job:  pytest tests/test_scaling_engines.py
"""

import numba
import numpy as np
import pytest

from mdx2.scaling import BatchModelRefiner, ScaledData
from mdx2.scaling_reference import BatchModelRefiner as RefBatchModelRefiner
from mdx2.scaling_reference import ScaledData as RefScaledData


def synthetic(n_groups=3000, mult=8, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * mult
    ih = np.repeat(np.arange(n_groups), mult)
    phi = rng.uniform(0, 40, n).astype(np.float32)
    ix = rng.uniform(0, 1000, n).astype(np.float32)
    iy = rng.uniform(0, 800, n).astype(np.float32)
    s = rng.uniform(0.05, 0.8, n).astype(np.float32)
    I0 = rng.exponential(200.0, n_groups)
    b = 1 + 0.2 * np.sin(phi / 7.0)
    d = 1 + 0.05 * np.cos(ix / 150.0) * np.sin(iy / 120.0)
    a = 1 + 0.03 * np.cos(phi / 11.0 + ix / 400.0)
    c = 20.0 * (1 + s)
    mu = a * d * (b * I0[ih] + c)
    I = rng.poisson(mu).astype(np.float32)
    sigma = np.sqrt(np.maximum(I, 1.0)).astype(np.float32)
    batch = (phi > 20).astype(int)  # two batches, contiguous after sorting
    order = np.argsort(batch, kind="stable")
    return {k: v[order] for k, v in dict(I=I, sigma=sigma, ih=ih, phi=phi, ix=ix, iy=iy, s=s, batch=batch).items()}


def build(cls_data, cls_ref, D):
    S = cls_data(D["I"], D["sigma"], D["ih"], phi=D["phi"], ix=D["ix"], iy=D["iy"], s=D["s"], batch=D["batch"])
    S.mask_singletons()
    MR = cls_ref(S)
    MR.add_scaling_models(dphi=1.0)
    MR.add_offset_models(dphi=5.0, ns=11)
    MR.add_absorption_models(dphi=10.0, nix=5, niy=5)
    MR.add_detector_model(nix=20, niy=20)
    return MR


def step(MR, kind):
    MR.apply()
    Im = MR.data.merge()[0]
    x2 = dict(
        b=lambda: MR.bfit(Im, 1.0),
        c=lambda: MR.cfit(Im, 1.0, 1.0, 0.001, 0.0),
        d=lambda: MR.dfit(Im, 1.0),
        a=lambda: MR.afit(Im, 10.0, 1.0),
    )[kind]()
    return float(x2)


@pytest.mark.parametrize("nthreads", [1, None])
def test_scaling_matches_reference(nthreads):
    if nthreads is not None:
        numba.set_num_threads(nthreads)
    else:
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    D = synthetic()
    ref = build(RefScaledData, RefBatchModelRefiner, D)
    fast = build(ScaledData, BatchModelRefiner, D)
    for kind in ("b", "c", "b", "c", "d", "a", "d", "a"):
        x2_ref, x2_fast = step(ref, kind), step(fast, kind)
        assert x2_fast >= 0
        assert x2_fast == pytest.approx(x2_ref, rel=1e-5), kind  # lsqr stops at 1e-6
    for k, (br_ref, br_fast) in enumerate(zip(ref.batch_refiners, fast.batch_refiners)):
        for name in ("scaling", "absorption", "offset", "detector"):
            u_ref = getattr(br_ref, name).model.u
            u_fast = getattr(br_fast, name).model.u
            scale = np.abs(u_ref).max()
            assert np.abs(u_fast - u_ref).max() <= 1e-3 * scale, (k, name)
    # outlier masking
    for MR in (ref, fast):
        MR.apply()
    n_ref = ref.data.mask_outliers(ref.data.merge()[0], 3.0)
    n_fast = fast.data.mask_outliers(fast.data.merge()[0], 3.0)
    assert n_fast == n_ref
