"""numba kernels for mdx2.scaling (passes over the observations; see mdx2.scaling for the algorithm).

The interpolation stencil of every observation is computed on the fly (same cell search, edge handling, corner order
and Fortran-order indexing as InterpLin1/2/3), so no interpolation matrices are stored. Kernels split the observations
into one chunk per thread and use per-thread accumulators (no atomics); scalars are kept in thread-local variables and
per-thread scratch rows are padded so threads never share a cache line.
"""

import math

import numba as nb
import numpy as np

PAD = 16  # per-thread scratch rows padded to >= 128 bytes so threads never share a cache line


@nb.njit(inline="always")
def _axis(x, lo, hi, w0, dw, N):
    """Cell and fraction along one axis (mdx2 InterpLin semantics); b = -1 if outside the grid."""
    if not (x >= lo and x <= hi):
        return -1, 0.0
    f = (x - w0) / dw
    b = int(math.floor(f))
    if b > N - 2:
        b = N - 2
    if b < 0:
        b = 0
    return b, f - b


@nb.njit(inline="always")
def _stencil(i, nd, x0, x1, x2, lo, hi, w0, dw, N, idx, wts, c):
    """Fill idx[c, :K], wts[c, :K] with observation i's interpolation stencil (corner order and Fortran-order
    indexing of mdx2's InterpLin1/2/3). Returns False if the observation is outside the grid."""
    b0, f0 = _axis(np.float64(x0[i]), lo[0], hi[0], w0[0], dw[0], N[0])
    if b0 < 0:
        return False
    b1, f1, b2, f2 = 0, 0.0, 0, 0.0
    if nd > 1:
        b1, f1 = _axis(np.float64(x1[i]), lo[1], hi[1], w0[1], dw[1], N[1])
        if b1 < 0:
            return False
    if nd > 2:
        b2, f2 = _axis(np.float64(x2[i]), lo[2], hi[2], w0[2], dw[2], N[2])
        if b2 < 0:
            return False
    for k in range(1 << nd):
        e0 = k & 1
        e1 = (k >> 1) & 1
        e2 = (k >> 2) & 1
        w = f0 if e0 else 1.0 - f0
        j = b0 + e0
        if nd > 1:
            w *= f1 if e1 else 1.0 - f1
            j += (b1 + e1) * N[0]
        if nd > 2:
            w *= f2 if e2 else 1.0 - f2
            j += (b2 + e2) * N[0] * N[1]
        idx[c, k] = j
        wts[c, k] = w
    return True


@nb.njit(parallel=True, cache=True, error_model="numpy")
def k_values(nd, x0, x1, x2, lo, hi, w0, dw, N, u, r0, r1, out, nchunk):
    """out[i] = interpolated u at observation i (0 outside the grid, as mdx2), rows r0..r1."""
    K = 1 << nd
    n = r1 - r0
    idx = np.zeros((nchunk, PAD), np.int64)
    wts = np.zeros((nchunk, PAD))
    for c in nb.prange(nchunk):
        for i in range(r0 + n * c // nchunk, r0 + n * (c + 1) // nchunk):
            if not _stencil(i, nd, x0, x1, x2, lo, hi, w0, dw, N, idx, wts, c):
                out[i] = 0.0
                continue
            acc = 0.0
            for k in range(K):
                acc += wts[c, k] * u[idx[c, k]]
            out[i] = acc


@nb.njit(parallel=True, cache=True, error_model="numpy")
def k_normal(kind, nd, x0, x1, x2, lo, hi, w0, dw, N, tab, r0, r1, mask, I, sigma, ih, Im, va, vb, vc, vd, nchunk):
    """One pass: A'A (neighbour slots), A'y, y'y and the count, for model `kind` with mdx2's weights:
    b: sigma' = sigma/(a d), y = (I/(a d) - c)/sigma', p = Im/sigma'
    a: p = d (b Im + c)/sigma,  y = I/sigma
    c: sigma' = sigma/(a d), p = 1/sigma', y = (I/(a d) - b Im)/sigma'
    d: p = a (b Im + c)/sigma,  y = I/sigma
    (A = diag(p) A0). Per-thread accumulators; scalars kept in locals and written once per thread."""
    K = 1 << nd
    nslot = 3**nd
    P = 1
    for d in range(nd):
        P *= N[d]
    n = r1 - r0
    S = np.zeros((nchunk, P, nslot))
    Ay = np.zeros((nchunk, P))
    yy = np.zeros((nchunk, PAD))
    cnt = np.zeros((nchunk, PAD), np.int64)
    idx = np.zeros((nchunk, PAD), np.int64)
    wts = np.zeros((nchunk, PAD))
    for c in nb.prange(nchunk):
        yy_c = 0.0
        cnt_c = 0
        for i in range(r0 + n * c // nchunk, r0 + n * (c + 1) // nchunk):
            if mask[i]:
                continue
            Ii = np.float64(I[i])
            si = np.float64(sigma[i])
            im = Im[ih[i]]
            if kind == 0:
                ad = va[i] * vd[i]
                sp = si / ad
                p = im / sp
                y = (Ii / ad - vc[i]) / sp
            elif kind == 1:
                p = vd[i] * (vb[i] * im + vc[i]) / si
                y = Ii / si
            elif kind == 2:
                ad = va[i] * vd[i]
                sp = si / ad
                p = 1.0 / sp
                y = (Ii / ad - vb[i] * im) / sp
            else:
                p = va[i] * (vb[i] * im + vc[i]) / si
                y = Ii / si
            cnt_c += 1
            yy_c += y * y
            if not _stencil(i, nd, x0, x1, x2, lo, hi, w0, dw, N, idx, wts, c):
                continue  # outside the grid: no model column (as mdx2: no entries in A0)
            p2 = p * p
            py = p * y
            for s in range(K):
                ws = wts[c, s]
                if ws == 0.0:
                    continue
                a = idx[c, s]
                Ay[c, a] += ws * py
                wsp = ws * p2
                for r in range(K):
                    S[c, a, tab[s, r]] += wsp * wts[c, r]
        yy[c, 0] = yy_c
        cnt[c, 0] = cnt_c
    out = np.zeros((P, nslot))
    aty = np.zeros(P)
    for a in nb.prange(P):
        for c in range(nchunk):
            aty[a] += Ay[c, a]
            for s in range(nslot):
                out[a, s] += S[c, a, s]
    return out, aty, yy[:, 0].sum(), cnt[:, 0].sum()


@nb.njit(cache=True)
def k_group_order(ih, G):
    """Observations ordered by group (stable counting sort, linear time) and each group's start in that order.
    Serial: runs before mdx2.scale starts numba's threads."""
    gstart = np.zeros(G + 1, np.int64)
    for i in range(ih.size):
        gstart[ih[i] + 1] += 1
    for h in range(G):
        gstart[h + 1] += gstart[h]
    pos = gstart[:-1].copy()
    order = np.empty(ih.size, np.int64)
    for i in range(ih.size):
        g = ih[i]
        order[pos[g]] = i
        pos[g] += 1
    return order, gstart


@nb.njit(parallel=True, cache=True, error_model="numpy")
def k_apply(va, vb, vc, vd, scale, offset, nchunk):
    """mdx2 apply: scale = a b d, offset = c / b (numpy semantics for b = 0)."""
    n = scale.size
    for c in nb.prange(nchunk):
        for i in range(n * c // nchunk, n * (c + 1) // nchunk):
            scale[i] = va[i] * vb[i] * vd[i]
            offset[i] = vc[i] / vb[i]


@nb.njit(parallel=True, cache=True, error_model="numpy")
def k_merge(gstart, order, mask, I, sigma, scale, offset, Im, sigm, counts):
    """mdx2 ScaledData.merge: I' = I/scale - offset, sigma' = sigma/scale, inverse-variance mean per group."""
    G = gstart.size - 1
    for h in nb.prange(G):
        num = 0.0
        den = 0.0
        m = 0
        for t in range(gstart[h], gstart[h + 1]):
            i = order[t]
            if not mask[i]:
                sc = scale[i]
                sp = np.float64(sigma[i]) / sc
                wi = 1.0 / (sp * sp)
                num += wi * (np.float64(I[i]) / sc - offset[i])
                den += wi
                m += 1
        counts[h] = m
        if m > 0:
            Im[h] = num / den
            sigm[h] = 1.0 / math.sqrt(den)
        else:
            Im[h] = np.nan
            sigm[h] = np.nan


@nb.njit(parallel=True, cache=True, error_model="numpy")
def k_outliers(mask, I, sigma, ih, Im, scale, offset, nsigma, nchunk):
    """mdx2: resid = (scale (Im + offset) - I)/sigma > nsigma (one-sided); masks them, returns the count."""
    n = mask.size
    cnt = np.zeros((nchunk, PAD), np.int64)
    for c in nb.prange(nchunk):
        cnt_c = 0
        for i in range(n * c // nchunk, n * (c + 1) // nchunk):
            if mask[i]:
                continue
            r = (scale[i] * (Im[ih[i]] + offset[i]) - np.float64(I[i])) / np.float64(sigma[i])
            if r > nsigma:
                mask[i] = True
                cnt_c += 1
        cnt[c, 0] = cnt_c
    return cnt[:, 0].sum()
