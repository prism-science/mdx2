"""Opt-in check of mdx2.scaling on real data: the first b-fit normal equations (A'A, A'y, y'y, count) and the
interpolated values must equal a direct numpy/scipy computation with mdx2's InterpLin1, per batch, with one thread
and with all available threads. Validates a machine / numba version / threading layer on large data, where problems
can appear that small tests miss.

Skipped unless MDX2_TEST_HKL is set to one or more corrected.nxs files (separated by os.pathsep):

    MDX2_TEST_HKL=mac1_4/corrected.nxs:mac1_5/corrected.nxs pytest -s tests/test_scaling_real_data.py
"""

import os
import time

import numba
import numpy as np
import pytest
from scipy import sparse

from mdx2._scaling_kernels import k_normal, k_values
from mdx2.command_line.scale import load_data_for_scaling
from mdx2.parallel import available_cpus
from mdx2.scaling import KIND_B, BatchModelRefiner, InterpLin1

FILES = [f for f in os.environ.get("MDX2_TEST_HKL", "").split(os.pathsep) if f]
pytestmark = pytest.mark.skipif(not FILES, reason="set MDX2_TEST_HKL to run on real data")


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def test_first_b_fit_on_real_data():
    S = load_data_for_scaling(*FILES)
    MR = BatchModelRefiner(S)
    MR.add_scaling_models(dphi=1.0)
    MR.apply()
    Im = MR.data.merge()[0]
    v = MR.vals
    nmax = min(available_cpus(), numba.config.NUMBA_NUM_THREADS)
    print(f"\nnumba {numba.__version__}, layer {numba.config.THREADING_LAYER}, {len(S):,} observations")
    for k, sl in enumerate(S._batches):
        t = MR.terms[k]["b"]
        g = t.grid
        m = ~S.mask[sl]
        I = S._I[sl][m].astype(np.float64)
        sig = S._sigma[sl][m].astype(np.float64)
        axis = np.linspace(g.lo[0], g.hi[0], g.N[0])
        A = sparse.diags(Im[S._ih[sl][m]] / sig) @ InterpLin1(S._phi[sl][m], axis).A.tocsr()
        y = I / sig
        AA_ref, Ay_ref, yy_ref = (A.T @ A).toarray(), A.T @ y, float(np.sum(y * y))
        for nch in (1, nmax):
            t0 = time.perf_counter()
            slots, Ay, yy, cnt = k_normal(
                KIND_B,
                t.nd,
                *t.x,
                g.lo,
                g.hi,
                g.lo,
                g.dw,
                g.N,
                g.tab,
                sl.start,
                sl.stop,
                S.mask,
                S._I,
                S._sigma,
                S._ih,
                Im,
                v["a"],
                v["b"],
                v["c"],
                v["d"],
                nch,
            )
            dt = time.perf_counter() - t0
            AA = g.to_csr(slots).toarray()
            print(
                f"batch {k}, {nch} chunk(s): {dt:.2f} s; rel err A'A {_rel(AA, AA_ref):.1e}, A'y {_rel(Ay, Ay_ref):.1e}"
                f", y'y {abs(yy - yy_ref) / yy_ref:.1e}; count {cnt:,} / {int(m.sum()):,}"
            )
            assert cnt == m.sum()
            assert _rel(AA, AA_ref) < 1e-10 and _rel(Ay, Ay_ref) < 1e-10 and abs(yy - yy_ref) < 1e-10 * yy_ref
        u = np.random.default_rng(k).normal(size=g.P)
        out = np.zeros(len(S))
        k_values(t.nd, *t.x, g.lo, g.hi, g.lo, g.dw, g.N, u, sl.start, sl.stop, out, nmax)
        assert np.abs(out[sl] - InterpLin1(S._phi[sl], axis).interp(u)).max() < 1e-5
