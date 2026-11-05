"""numba kernels for mdx2.integration (per-chunk binning of pixels on the Miller-index grid).

One pass over a chunk: trilinear h, k, l at each pixel (same weights and operation order as
mdx2.utils.interp_g2g_trilinear, so the indices are bit-identical to MillerIndex.regrid), the peak mask, and sums per
subdivided voxel (as HKLTable.bin over ImageSeries.index's table). Serial and GIL-free: callers parallelize over
chunks (processes or threads).
"""

import math

import numba as nb
import numpy as np


@nb.njit(cache=True, nogil=True, inline="always")
def _interp(v, a, b, c, fa, fb, fc):
    """Trilinear value of grid v (axes in mdx2's sorted order) at base (a, b, c) + fractions, with
    _trilinear_g2g's formula and operation order."""
    c00 = v[a, b, c] * (1 - fa) + v[a + 1, b, c] * fa
    c01 = v[a, b, c + 1] * (1 - fa) + v[a + 1, b, c + 1] * fa
    c10 = v[a, b + 1, c] * (1 - fa) + v[a + 1, b + 1, c] * fa
    c11 = v[a, b + 1, c + 1] * (1 - fa) + v[a + 1, b + 1, c + 1] * fa
    c0 = c00 * (1 - fb) + c10 * fb
    c1 = c01 * (1 - fb) + c11 * fb
    return c0 * (1 - fc) + c1 * fc


@nb.njit(cache=True, nogil=True, inline="always")
def _rint(x):
    """Round half to even, as np.round."""
    r = math.floor(x + 0.5)
    if r - x == 0.5 and (r % 2) != 0:
        r -= 1.0
    return r


@nb.njit(cache=True, nogil=True)
def bin_chunk_kernel(
    data, maskval, static_mask, use_static, use_peak, bragg,
    b0, f0, b1, f1, b2, f2, order,
    Hg, Kg, Lg,
    phi, iy, ix, seconds,
    W, r0, cut2, allowed, a_lo,
    ndiv, box_lo, box_shape, row_mode,
):  # fmt: skip
    """Sums per voxel of a dense box: pixels, phi, iy, ix, counts, seconds.

    data: chunk counts (maskval = invalid). Interpolation: grid nodes Hg, Kg, Lg (axes in sorted order, see
    mdx2.integration.bin_chunk), per-axis bases b* and fractions f* in that order; order[j] = chunk axis of sorted
    axis j. Peak mask (use_peak): inside = |W (dh - r0)|^2 < cut2 (as peak_expression), allowed = reflection-
    condition lookup at round(h) - a_lo; excluded = bragg XOR (allowed and inside). Voxel = round(h * ndiv) -
    box_lo. row_mode (ix innermost and last combined axis): first two interpolation stages once per row and ix node.
    """
    nbox = box_shape[0] * box_shape[1] * box_shape[2]
    pixels = np.zeros(nbox, np.int64)
    s_phi = np.zeros(nbox)
    s_iy = np.zeros(nbox)
    s_ix = np.zeros(nbox)
    s_counts = np.zeros(nbox)
    s_sec = np.zeros(nbox)
    idx = np.empty(3, np.int64)
    n_p, n_y, n_x = data.shape
    nn = Hg.shape[2]
    Ph = np.empty(nn)
    Pk = np.empty(nn)
    Pl = np.empty(nn)
    for i in range(n_p):
        for j in range(n_y):
            if row_mode:
                idx[0] = i
                idx[1] = j
                ia = idx[order[0]]
                ib = idx[order[1]]
                a = b0[ia]
                b = b1[ib]
                fa = f0[ia]
                fb = f1[ib]
                for n in range(nn):
                    # second-stage value at node n: the reference's c0 for base n and c1 for base n - 1
                    Ph[n] = (Hg[a, b, n] * (1 - fa) + Hg[a + 1, b, n] * fa) * (1 - fb) + (
                        Hg[a, b + 1, n] * (1 - fa) + Hg[a + 1, b + 1, n] * fa
                    ) * fb
                    Pk[n] = (Kg[a, b, n] * (1 - fa) + Kg[a + 1, b, n] * fa) * (1 - fb) + (
                        Kg[a, b + 1, n] * (1 - fa) + Kg[a + 1, b + 1, n] * fa
                    ) * fb
                    Pl[n] = (Lg[a, b, n] * (1 - fa) + Lg[a + 1, b, n] * fa) * (1 - fb) + (
                        Lg[a, b + 1, n] * (1 - fa) + Lg[a + 1, b + 1, n] * fa
                    ) * fb
            for k in range(n_x):
                cval = data[i, j, k]
                if cval == maskval:
                    continue
                if use_static and static_mask[i, j, k]:
                    continue
                if row_mode:
                    c = b2[k]
                    fc = f2[k]
                    h = Ph[c] * (1 - fc) + Ph[c + 1] * fc
                    kk = Pk[c] * (1 - fc) + Pk[c + 1] * fc
                    ll = Pl[c] * (1 - fc) + Pl[c + 1] * fc
                else:
                    idx[0] = i
                    idx[1] = j
                    idx[2] = k
                    ia = idx[order[0]]
                    ib = idx[order[1]]
                    ic = idx[order[2]]
                    h = _interp(Hg, b0[ia], b1[ib], b2[ic], f0[ia], f1[ib], f2[ic])
                    kk = _interp(Kg, b0[ia], b1[ib], b2[ic], f0[ia], f1[ib], f2[ic])
                    ll = _interp(Lg, b0[ia], b1[ib], b2[ic], f0[ia], f1[ib], f2[ic])
                if use_peak:
                    e0 = h - math.floor(h + 0.5) - r0[0]
                    e1 = kk - math.floor(kk + 0.5) - r0[1]
                    e2 = ll - math.floor(ll + 0.5) - r0[2]
                    d0 = W[0, 0] * e0 + W[0, 1] * e1 + W[0, 2] * e2
                    d1 = W[1, 0] * e0 + W[1, 1] * e1 + W[1, 2] * e2
                    d2 = W[2, 0] * e0 + W[2, 1] * e1 + W[2, 2] * e2
                    m = (d0 * d0 + d1 * d1 + d2 * d2) < cut2
                    if m:
                        m = allowed[int(_rint(h)) - a_lo[0], int(_rint(kk)) - a_lo[1], int(_rint(ll)) - a_lo[2]]
                    if m != bragg:  # excluded = bragg XOR (allowed and inside)
                        continue
                bx = (int(_rint(h * ndiv[0])) - box_lo[0]) * box_shape[1]
                bx = (bx + int(_rint(kk * ndiv[1])) - box_lo[1]) * box_shape[2]
                v = bx + int(_rint(ll * ndiv[2])) - box_lo[2]
                pixels[v] += 1
                s_phi[v] += phi[i]
                s_iy[v] += iy[j]
                s_ix[v] += ix[k]
                s_counts[v] += cval
                s_sec[v] += seconds[i]
    return pixels, s_phi, s_iy, s_ix, s_counts, s_sec
