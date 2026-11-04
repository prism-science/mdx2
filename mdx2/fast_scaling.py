"""Compiled, multi-threaded engine for mdx2.scale (same algorithm as mdx2.scaling's ALS).

Drop-in replacements for ScaledData and BatchModelRefiner, selected with `mdx2.scale --engine fast`. Models, grids,
penalties, the lambda recipe (alpha tr(A'A)/tr(B'B)), the lsqr solve, chi2, the offset threshold / level shift,
outlier masking and the stage schedule are mdx2's. What differs is how the passes over the observations are done:

- the interpolation stencil of every observation is computed on the fly inside the kernels (same cell search, edge
  handling, corner order and Fortran-order indexing as InterpLin1/2/3), so no interpolation matrices are stored;
- A'A, A'y and y'y for a fit are accumulated in ONE threaded pass, with mdx2's per-model weights and targets computed
  inline; A'A is accumulated as neighbour slots on the control-point grid (each stencil only touches adjacent points);
- merging, outlier and singleton masking are threaded kernels; masked observations get zero weight (a plain boolean
  mask, no masked-array copies).

Requires numba. Single-process; threads via numba (set with --nproc).
"""

import math

import numba as nb
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from mdx2.profiling import note

try:  # lsqr's vector operations are tiny: keep BLAS single-threaded (no oversubscription with numba's threads)
    from threadpoolctl import threadpool_limits
except ImportError:  # optional
    threadpool_limits = None
from mdx2.scaling import (
    AbsorptionModel,
    DetectorModel,
    InterpLin1,
    InterpLin2,
    InterpLin3,
    OffsetModel,
    ScalingModel,
)

KIND_B, KIND_A, KIND_C, KIND_D = 0, 1, 2, 3


def _check(what, got, expected):
    """Guard against a threading runtime that executes parallel-loop iterations more than once (seen with numba
    0.65.1's GNU OpenMP layer on a 96-core Linux node: inflated sums, negative chi2)."""
    if got != expected:
        try:
            layer = nb.threading_layer()
        except ValueError:  # not initialized yet
            layer = nb.config.THREADING_LAYER
        raise RuntimeError(
            f"fast scaling engine: {what} counted {got} observations, expected {expected}. The numba threading layer "
            f"({layer}) is not executing parallel loops correctly on this machine; run with "
            f"NUMBA_THREADING_LAYER=workqueue (the fast engine's default) or --engine default."
        )


# ---- kernels ------------------------------------------------------------------------------------------------------


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


# ---- grid description ---------------------------------------------------------------------------------------------


class _Grid:
    """Uniform control-point grid of an mdx2 model (axes as in the model, Fortran order)."""

    def __init__(self, axes):
        self.axes = [np.asarray(a, dtype=np.float64) for a in axes]
        for a in self.axes:
            assert a.size >= 2, "mdx2 interpolation needs at least two control points per axis"
            assert np.allclose(np.diff(a), (a[-1] - a[0]) / (a.size - 1), rtol=1e-9, atol=1e-12), "non-uniform axis"
        self.lo = np.array([a[0] for a in self.axes])
        self.hi = np.array([a[-1] for a in self.axes])
        self.N = np.array([a.size for a in self.axes], dtype=np.int64)
        self.dw = (self.hi - self.lo) / (self.N - 1)
        self.P = int(np.prod(self.N))
        nd = self.N.size
        K = 1 << nd
        tab = np.zeros((K, K), dtype=np.int64)
        for s in range(K):
            for r in range(K):
                tab[s, r] = sum((((r >> d) & 1) - ((s >> d) & 1) + 1) * 3**d for d in range(nd))
        self.tab = tab
        strides = np.cumprod([1] + list(self.N[:-1]))
        offs = np.array(np.unravel_index(np.arange(3**nd), [3] * nd, order="F")).T - 1
        self.delta = offs @ strides

    def to_csr(self, slots):
        rows = np.repeat(np.arange(self.P), slots.shape[1])
        cols = rows + np.tile(self.delta, self.P)
        v = slots.ravel()
        ok = v != 0
        return sparse.csr_matrix((v[ok], (rows[ok], cols[ok])), shape=(self.P, self.P))


def _calc_points(x, dx=1, nx=None, integer_limits=True, x_min=None, x_max=None):
    """mdx2 _ModelRefiner._calc_points."""
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    if integer_limits:
        x_min = np.floor(x_min)
        x_max = np.ceil(x_max)
    if nx is None:
        nx = np.round((x_max - x_min) / dx).astype(int) + 1
    return np.linspace(x_min, x_max, nx)


# ---- data ---------------------------------------------------------------------------------------------------------


class FastScaledData:
    """Observations for scaling (same constructor and the methods mdx2.scale uses as mdx2.scaling.ScaledData)."""

    def __init__(
        self, I, sigma, ih, phi=None, ix=None, iy=None, s=None, batch=None, scale=None, offset=None, mask=None
    ):
        n = len(I)
        self._I, self._sigma = np.ascontiguousarray(I), np.ascontiguousarray(sigma)
        self._ih = np.ascontiguousarray(ih, dtype=np.int64)
        self._phi, self._ix, self._iy, self._s = (
            np.ascontiguousarray(a) if a is not None else None for a in (phi, ix, iy, s)
        )
        if batch is None:
            self._batches = [slice(0, n)]
        else:
            batch = np.asarray(batch)
            edges = np.flatnonzero(np.diff(batch)) + 1
            assert np.all(np.diff(batch) >= 0), "observations must be grouped by batch"
            bounds = np.r_[0, edges, n]
            self._batches = [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        self.scale = np.ones(n) if scale is None else np.asarray(scale, dtype=np.float64)
        self.offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
        if mask is None:
            sg = self._sigma
            mask = np.isnan(self._I) | np.isnan(sg) | (sg == 0) | np.isinf(sg)
        self.mask = np.ascontiguousarray(mask, dtype=np.bool_)
        self._ihmax = int(self._ih.max())
        G = self._ihmax + 1
        self._order = np.argsort(self._ih, kind="stable")
        self._gstart = np.searchsorted(self._ih[self._order], np.arange(G + 1)).astype(np.int64)

    @property
    def nchunk(self):
        """Work chunks for the kernels: numba's current thread count (set by mdx2.scale after loading)."""
        return nb.get_num_threads()

    def __len__(self):
        return len(self._I)

    @property
    def nbatches(self):
        return len(self._batches)

    def mask_singletons(self):
        n = np.bincount(self._ih[~self.mask], minlength=self._ihmax + 1)
        single = (n <= 1)[self._ih] & ~self.mask
        self.mask |= single
        return int(single.sum())

    def merge(self):
        G = self._ihmax + 1
        Im, sigm, counts = np.empty(G), np.empty(G), np.empty(G, dtype=np.int64)
        k_merge(self._gstart, self._order, self.mask, self._I, self._sigma, self.scale, self.offset, Im, sigm, counts)
        _check("merge", int(counts.sum()), self.count())
        return Im, sigm, counts

    def mask_outliers(self, Im, nsigma):
        return int(
            k_outliers(
                self.mask,
                self._I,
                self._sigma,
                self._ih,
                np.asarray(Im, dtype=np.float64),
                self.scale,
                self.offset,
                float(nsigma),
                self.nchunk,
            )
        )

    def count(self, sl=slice(None)):
        return int(np.count_nonzero(~self.mask[sl]))


# ---- refiner ------------------------------------------------------------------------------------------------------


class _Term:
    """One model (b, a, c or d) on one batch (or all data for d)."""

    def __init__(self, model, coords, rows, penalties):
        self.model = model
        self.grid = _Grid(self._axes(model))
        self.coords = coords  # tuple of coordinate arrays (full length)
        self.nd = len(coords)
        self.x = tuple(coords) + (coords[0],) * (3 - len(coords))  # x0, x1, x2 (unused axes: placeholders)
        self.rows = rows  # slice of observations
        self.BB = [(B.T @ B).tocsr() for B in penalties]

    @staticmethod
    def _axes(m):
        if isinstance(m, ScalingModel):
            return [m.x]
        if isinstance(m, AbsorptionModel):
            return [m.x, m.y, m.z]
        return [m.x, m.y]  # OffsetModel (s, phi), DetectorModel (ix, iy)


class _BatchView:
    """What mdx2.scale's saving loop reads from each batch refiner."""

    class _M:
        def __init__(self, model):
            self.model = model

    def __init__(self, terms):
        self.scaling, self.absorption, self.offset, self.detector = (
            self._M(terms.get(k)) for k in ("b", "a", "c", "d")
        )


class FastBatchModelRefiner:
    def __init__(self, data):
        self.data = data
        nb_ = data.nbatches
        self.terms = [dict() for _ in range(nb_)]  # per batch: "b", "a", "c" -> _Term
        self.dterm = None
        n = len(data)
        self.vals = dict(a=np.ones(n), b=np.ones(n), c=np.zeros(n), d=np.ones(n))

    # -- models (control points as mdx2: from the unmasked data of each batch; detector from all data) --

    def _unmasked(self, arr, sl):
        return arr[sl][~self.data.mask[sl]]

    def add_scaling_models(self, dphi=1, nearest_degree=True, nphi=None):
        D = self.data
        for k, sl in enumerate(D._batches):
            m = ScalingModel(_calc_points(self._unmasked(D._phi, sl), dx=dphi, integer_limits=nearest_degree, nx=nphi))
            L = InterpLin1(m.x[:1], m.x)
            self._add(k, "b", m, (D._phi,), sl, [L.B], init=1.0)

    def add_offset_models(self, dphi=2.5, nearest_degree=True, nphi=None, ns=31, ds=None):
        D = self.data
        for k, sl in enumerate(D._batches):
            phi = _calc_points(self._unmasked(D._phi, sl), dx=dphi, integer_limits=nearest_degree, nx=nphi)
            s = _calc_points(self._unmasked(D._s, sl), dx=ds, integer_limits=False, nx=ns)
            m = OffsetModel(s, phi)
            L = InterpLin2(m.x[:1], m.y[:1], m.x, m.y)
            self._add(k, "c", m, (D._s, D._phi), sl, [L.Bx, L.By], init=0.0)

    def add_absorption_models(self, dphi=10, nearest_degree=True, nphi=None, nix=20, dix=None, niy=20, diy=None):
        D = self.data
        for k, sl in enumerate(D._batches):
            phi = _calc_points(self._unmasked(D._phi, sl), dx=dphi, integer_limits=nearest_degree, nx=nphi)
            ix = _calc_points(self._unmasked(D._ix, sl), dx=dix, integer_limits=True, nx=nix)
            iy = _calc_points(self._unmasked(D._iy, sl), dx=diy, integer_limits=True, nx=niy)
            m = AbsorptionModel(ix, iy, phi)
            L = InterpLin3(m.x[:1], m.y[:1], m.z[:1], m.x, m.y, m.z)
            self._add(k, "a", m, (D._ix, D._iy, D._phi), sl, [L.Bxy, L.Bz], init=1.0)

    def add_detector_model(self, nix=20, dix=None, niy=20, diy=None):
        D = self.data
        allrows = slice(0, len(D))
        ix = _calc_points(self._unmasked(D._ix, allrows), dx=dix, integer_limits=True, nx=nix)
        iy = _calc_points(self._unmasked(D._iy, allrows), dx=diy, integer_limits=True, nx=niy)
        m = DetectorModel(ix, iy)
        L = InterpLin2(m.x[:1], m.y[:1], m.x, m.y)
        self.dterm = _Term(m, (D._ix, D._iy), allrows, [L.B])
        m.set_u(np.ones(self.dterm.grid.P))
        self._update_values("d", self.dterm)

    def _add(self, k, name, model, coords, sl, penalties, init):
        t = _Term(model, coords, sl, penalties)
        model.set_u(np.full(t.grid.P, init))
        self.terms[k][name] = t
        self._update_values(name, t)

    def _update_values(self, name, t):
        g = t.grid
        u = np.ascontiguousarray(t.model.u.ravel(order="F"), dtype=np.float64)
        k_values(
            t.nd, *t.x, g.lo, g.hi, g.lo, g.dw, g.N, u, t.rows.start, t.rows.stop, self.vals[name], self.data.nchunk
        )

    @property
    def batch_refiners(self):
        return iter(
            _BatchView({k: t.model for k, t in dict(terms, d=self.dterm).items() if t is not None})
            for terms in self.terms
        )

    # -- apply / fits ------------------------------------------------------------------------------------------------

    def apply(self):
        v = self.vals
        k_apply(v["a"], v["b"], v["c"], v["d"], self.data.scale, self.data.offset, self.data.nchunk)

    def _fit(self, kind, t, Im, alphas, ridge=None):
        D, v, g = self.data, self.vals, t.grid
        slots, Ay, yy, count = k_normal(
            kind,
            t.nd,
            *t.x,
            g.lo,
            g.hi,
            g.lo,
            g.dw,
            g.N,
            g.tab,
            t.rows.start,
            t.rows.stop,
            D.mask,
            D._I,
            D._sigma,
            D._ih,
            np.asarray(Im, dtype=np.float64),
            v["a"],
            v["b"],
            v["c"],
            v["d"],
            D.nchunk,
        )
        _check("fit", int(count), D.count(t.rows))
        AA = g.to_csr(slots)
        trAA = AA.diagonal().sum()
        G = AA.copy()
        for a_mult, BB in zip(alphas, t.BB):
            G = G + (a_mult * trAA / BB.diagonal().sum()) * BB
        if ridge is not None:  # mdx2 offset: alpha_c * max(diag(A'A)) * I
            G = G + ridge * np.max(AA.diagonal()) * sparse.eye(g.P)
        if threadpool_limits is not None:
            with threadpool_limits(limits=1, user_api="blas"):
                res = lsqr(G, Ay)
        else:
            res = lsqr(G, Ay)
        u = res[0]
        note(lsqr_iters=int(res[2]), n_params=int(g.P), nnz_G=int(G.nnz))
        x2 = (u @ (AA @ u) - 2 * u @ Ay + yy) / count
        return u, x2

    def _set(self, name, t, u):
        t.model.set_u(u)
        self._update_values(name, t)

    def bfit(self, Imerge, alpha_mult):
        x2 = []
        for terms in self.terms:
            t = terms["b"]
            u, x = self._fit(KIND_B, t, Imerge, [alpha_mult])
            self._set("b", t, u)
            x2.append(x)
        return np.mean(x2)

    def afit(self, Imerge, alpha_xy_mult, alpha_z_mult):
        x2 = []
        for terms in self.terms:
            t = terms["a"]
            u, x = self._fit(KIND_A, t, Imerge, [alpha_xy_mult, alpha_z_mult])
            self._set("a", t, u)
            x2.append(x)
        return np.mean(x2)

    def cfit(self, Imerge, alpha_x_mult, alpha_y_mult, alpha_c_mult, min_c):
        x2 = []
        for terms in self.terms:
            t = terms["c"]
            u, x = self._fit(KIND_C, t, Imerge, [alpha_x_mult, alpha_y_mult], ridge=alpha_c_mult)
            if min_c is not None:  # mdx2: threshold, then level shift
                u[u < min_c] = min_c
                u += min_c - np.min(u)
            self._set("c", t, u)
            x2.append(x)
        return np.mean(x2)

    def dfit(self, Imerge, alpha_mult):
        u, x2 = self._fit(KIND_D, self.dterm, Imerge, [alpha_mult])
        self._set("d", self.dterm, u)
        return x2
