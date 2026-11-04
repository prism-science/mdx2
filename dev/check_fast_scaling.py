"""Diagnose the fast scaling engine on real data: compare its first b-fit normal equations and interpolated values
with a direct numpy/scipy computation (mdx2's InterpLin1), per batch, single- and multi-threaded, with timings.

usage: python dev/check_fast_scaling.py file1.nxs [file2.nxs ...] [--nproc 64]
"""

import argparse
import time

import numba
import numpy as np
from scipy import sparse

from mdx2.command_line.scale import load_data_for_scaling
from mdx2.fast_scaling import KIND_B, FastBatchModelRefiner, k_normal, k_values
from mdx2.scaling import InterpLin1

ap = argparse.ArgumentParser()
ap.add_argument("hkl", nargs="+")
ap.add_argument("--nproc", type=int, default=numba.config.NUMBA_NUM_THREADS)
a = ap.parse_args()
numba.set_num_threads(a.nproc)
print(f"numba {numba.__version__}, threads {numba.get_num_threads()}, layer {numba.config.THREADING_LAYER}")

t0 = time.perf_counter()
S = load_data_for_scaling(*a.hkl, engine="fast")
print(
    f"load + group + init: {time.perf_counter() - t0:.1f} s, {len(S):,} obs, "
    f"batches {[(b.start, b.stop) for b in S._batches]}"
)
MR = FastBatchModelRefiner(S)
t0 = time.perf_counter()
MR.add_scaling_models(dphi=1.0)
print(
    f"add_scaling_models (incl. k_values compile): {time.perf_counter() - t0:.1f} s; "
    f"threading layer used: {numba.threading_layer()}"
)
MR.apply()
t0 = time.perf_counter()
Im = MR.data.merge()[0]
print(f"merge: {time.perf_counter() - t0:.2f} s")


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


for k, sl in enumerate(S._batches):
    t = MR.terms[k]["b"]
    g = t.grid
    m = ~S.mask[sl]
    I = S._I[sl][m].astype(np.float64)
    sig = S._sigma[sl][m].astype(np.float64)
    print(
        f"\nbatch {k}: rows {sl.start:,}-{sl.stop:,}, unmasked {m.sum():,}, "
        f"b grid {g.N[0]} points [{g.lo[0]}, {g.hi[0]}], "
        f"|I/sigma| max {np.abs(I / sig).max():.3e}, sigma min {sig.min():.3e}"
    )
    # reference (mdx2 formulas, a = d = 1, c = 0 at the start): p = Im/sigma, y = I/sigma; A = diag(p) A0
    A0 = InterpLin1(S._phi[sl][m], g.axes[0] if hasattr(g, "axes") else np.linspace(g.lo[0], g.hi[0], g.N[0])).A.tocsr()
    p = Im[S._ih[sl][m]] / sig
    y = I / sig
    A = sparse.diags(p) @ A0
    AA_ref = (A.T @ A).toarray()
    Ay_ref = A.T @ y
    yy_ref = float(np.sum(y * y))
    v = MR.vals
    for nch in (1, numba.get_num_threads()):
        for rep in ("first", "repeat"):
            t1 = time.perf_counter()
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
            dt = time.perf_counter() - t1
        AA = g.to_csr(slots).toarray()
        print(
            f"  k_normal nchunk {nch:3d}: {dt:.2f} s (repeat call) | "
            f"rel err AA {rel(AA, AA_ref):.1e}, Ay {rel(Ay, Ay_ref):.1e}, "
            f"yy {abs(yy - yy_ref) / yy_ref:.1e}, count {cnt:,} (ref {m.sum():,})"
        )
    # interpolation check with random u
    u = np.random.default_rng(k).normal(size=g.P)
    out = np.zeros(len(S))
    k_values(t.nd, *t.x, g.lo, g.hi, g.lo, g.dw, g.N, u, sl.start, sl.stop, out, numba.get_num_threads())
    ref = InterpLin1(S._phi[sl], np.linspace(g.lo[0], g.hi[0], g.N[0])).interp(u)
    print(f"  k_values vs InterpLin1.interp: max abs diff {np.abs(out[sl] - ref).max():.1e}")
