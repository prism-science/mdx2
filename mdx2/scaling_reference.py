"""Reference implementation of mdx2.scale's ALS (the scaling engine of mdx2 <= 1.0.5).

Kept for one release, selectable with `mdx2.scale --engine reference`, and as the reference in the tests. Same
algorithm as mdx2.scaling's ScaledData / BatchModelRefiner; slower (rebuilds sparse interpolation matrices for every
fit, single-threaded) and needs more memory. Also used by mdx2.merge until it moves to the new classes.
"""

import copy

import numpy as np
from scipy import sparse

from mdx2.profiling import note
from mdx2.scaling import AbsorptionModel, DetectorModel, OffsetModel, ScalingModel  # noqa: F401


class ScaledData:
    def __init__(
        self, I, sigma, ih, phi=None, ix=None, iy=None, s=None, batch=None, scale=None, offset=None, mask=None
    ):
        if scale is None:
            scale = np.ones_like(I)
        if offset is None:
            offset = np.zeros_like(I)
        if mask is None:
            mask = np.isnan(I) | np.isnan(sigma) | (sigma == 0) | np.isinf(sigma)
        if batch is None:
            self._batches = [slice(0, len(I) + 1)]
        else:  # batch is a sorted list of integers
            if not np.all(batch[:-1] <= batch[1:]):  # must be sorted
                raise ValueError
            breakpoints = [0] + list(np.nonzero(np.diff(batch))[0] + 1) + [len(I) + 1]
            starts = breakpoints[:-1]
            stops = breakpoints[1:]
            self._batches = [slice(start, stop) for start, stop in zip(starts, stops)]
            # note: batch number is not saved! they are just stored sequentially in the list

        if phi is None:
            phi = np.full_like(I, np.nan)
        if ix is None:
            ix = np.full_like(I, np.nan)
        if iy is None:
            iy = np.full_like(I, np.nan)
        if s is None:
            s = np.full_like(I, np.nan)

        self._I = I
        self._sigma = sigma
        self._ih = ih.astype(int)
        self._phi = phi
        self._ix = ix
        self._iy = iy
        self._s = s
        self.scale = scale
        self.offset = offset
        self.mask = mask
        self._ihmax = np.max(ih)

    def __getitem__(self, ind):
        dset = ScaledData(
            self._I[ind],
            self._sigma[ind],
            self._ih[ind],
            phi=self._phi[ind],
            ix=self.ix[ind],
            iy=self.iy[ind],
            s=self.s[ind],
            scale=self.scale[ind],
            offset=self.offset[ind],
            mask=self.mask[ind],
        )
        dset._ihmax = self._ihmax
        # print('DEBUG: sliced dataset. Is scale a copy?',dset.scale.base is None)
        return dset

    def batches(self):
        """generator that loops over batches, returning a ScaledData object for each"""
        for sl in self._batches:
            yield self[sl]

    @property
    def nbatches(self):
        return len(self._batches)

    def _masked(self, arr):
        return np.ma.MaskedArray(data=np.ma.getdata(arr), mask=self.mask, copy=False)

    @property
    def I(self):  # noqa: E743
        # suppress divide-by-zero warning
        old_settings = np.seterr(all="ignore")
        val = self._masked(self._I / self.scale - self.offset)
        np.seterr(**old_settings)
        return val

    @property
    def sigma(self):
        old_settings = np.seterr(all="ignore")
        val = self._masked(self._sigma / self.scale)
        np.seterr(**old_settings)
        return val

    @property
    def ih(self):
        return self._masked(self._ih)

    @property
    def phi(self):
        return self._masked(self._phi)

    @property
    def ix(self):
        return self._masked(self._ix)

    @property
    def iy(self):
        return self._masked(self._iy)

    @property
    def s(self):
        return self._masked(self._s)

    def predict(self, Imerge):
        return self._masked(self.scale * (Imerge[self._ih] + self.offset))

    def copy(self):
        SD = copy.copy(self)
        SD.scale = copy.copy(self.scale)
        SD.offset = copy.copy(self.offset)
        SD.mask = copy.copy(self.mask)
        return SD

    def merge(self):
        n = self._ihmax + 1
        w = 1 / self.sigma.compressed() ** 2
        Iw = self.I.compressed() * w
        ind = self.ih.compressed()
        counts = np.bincount(ind, minlength=n)
        wm = np.bincount(ind, weights=w, minlength=n)
        Im = np.bincount(ind, weights=Iw, minlength=n)
        wm = np.ma.masked_where(counts == 0, wm, copy=False)
        Im = np.ma.masked_where(counts == 0, Im, copy=False)
        Im = Im / wm
        sigmam = 1 / np.sqrt(wm)
        return Im, sigmam, counts

    def mask_singletons(self):
        """Mask out reflections that only have one observation"""
        n = self._ihmax + 1
        counts = np.bincount(self._ih, minlength=n, weights=~self.mask)
        singletons = counts <= 1
        is_singleton = singletons[self._ih] & ~(self.mask)
        self.mask[is_singleton] = True  # apply singleton masks
        return is_singleton.sum()

    def mask_outliers(self, Im, nsigma, method="legacy"):
        if method != "legacy":
            raise ValueError("the reference engine only implements the legacy outlier rejection")
        resid = (self.scale * (Im[self._ih] + self.offset) - self._I) / self._sigma
        isoutlier = (resid > nsigma) & ~(self.mask)
        self.mask[isoutlier] = True  # apply outlier masks
        return isoutlier.sum()


class _ModelRefiner:
    def __init__(self, data, model=None):
        self.data = data
        self.model = model
        self._value = None  # cached value of the interpolated model at datapoints

    def calc_scale_offset(self, a=None, b=None, c=None, d=None):
        scale = np.ones_like(self.data._I, dtype=np.double)
        offset = np.zeros_like(scale)
        if a is None:
            a = 1.0
        if b is None:
            b = 1.0
        if c is None:
            c = 0.0
        if d is None:
            d = 1.0
        scale *= a * b * d
        offset += c / b
        return scale, offset

    def _calc_points(self, x, dx=1, nx=None, integer_limits=True, x_min=None, x_max=None):
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

    def add_model(self, *args, **kwargs):
        raise NotImplementedError

    def calc_problem(self, Imerge, **kwargs):
        raise NotImplementedError

    def fit(self, Imerge, **kwargs):
        raise NotImplementedError

    @property
    def value(self):
        if self._value is None:
            self._value = self.calc_value()
        return self._value

    def refine(self, Imerge, *args, **kwargs):
        ufit, x2 = self.fit(Imerge, *args, **kwargs)
        self.model.set_u(ufit)
        self._value = None  # trigger re-calculation when requested
        return x2


class ScalingModelRefiner(_ModelRefiner):
    def add_model(self, dphi=1, nearest_degree=True, nphi=None):
        phi_points = self._calc_points(self.data.phi, dx=dphi, integer_limits=nearest_degree, nx=nphi)
        self.model = ScalingModel(phi_points)
        return self

    def calc_value(self):
        if self.model is None:
            return 1.0
        return self.model.interp(self.data._phi)

    def calc_problem(self, Imerge, a=None, c=None, d=None):
        A0, B = self.model.interp_matrices(
            self.data.phi.compressed(),
        )

        # compute predicted intensities a = b = d = 1, c = 0
        tmp = self.data.copy()
        tmp.scale, tmp.offset = self.calc_scale_offset()
        Ipred = tmp.predict(Imerge).compressed()

        # compute scaled intensities with b = 1
        tmp.scale, tmp.offset = self.calc_scale_offset(a=a, c=c, d=d)
        I = tmp.I.compressed()
        sigma = tmp.sigma.compressed()

        # compute fitting matrices
        A = sparse.diags(Ipred / sigma) @ A0
        y = I / sigma
        AA = A.T @ A
        BB = B.T @ B
        Ay = A.T @ y
        yy = np.sum(y**2)
        return AA, BB, Ay, yy

    def fit(self, Imerge, alpha_mult, a=None, c=None, d=None):
        AA, BB, Ay, yy = self.calc_problem(Imerge, a=a, c=c, d=d)
        alpha = alpha_mult * AA.trace() / BB.trace()
        G = AA + alpha * BB
        if sparse.issparse(G):
            res = sparse.linalg.lsqr(G, Ay)
            ufit = res[0]
            note(lsqr_iters=int(res[2]), n_params=int(G.shape[0]), nnz_G=int(G.nnz))
        else:
            ufit = np.linalg.lstsq(G, Ay, rcond=None)[0]
        x2 = ufit @ AA @ ufit - 2 * ufit @ Ay + yy
        x2 = x2 / self.data.ih.count()
        return ufit, x2


class AbsorptionModelRefiner(_ModelRefiner):
    def add_model(self, dphi=10, nearest_degree=True, nphi=None, nix=20, dix=None, niy=20, diy=None):
        phi_points = self._calc_points(self.data.phi, dx=dphi, integer_limits=nearest_degree, nx=nphi)
        ix_points = self._calc_points(self.data.ix, dx=dix, integer_limits=True, nx=nix)
        iy_points = self._calc_points(self.data.iy, dx=diy, integer_limits=True, nx=niy)
        self.model = AbsorptionModel(ix_points, iy_points, phi_points)

    def calc_value(self):
        if self.model is None:
            return 1.0
        return self.model.interp(self.data._ix, self.data._iy, self.data._phi)

    def calc_problem(self, Imerge, b=None, c=None, d=None):
        A0, Bxy, Bz = self.model.interp_matrices(
            self.data.ix.compressed(),
            self.data.iy.compressed(),
            self.data.phi.compressed(),
        )

        # compute predicted intensities a = 1
        tmp = self.data.copy()
        tmp.scale, tmp.offset = self.calc_scale_offset(b=b, c=c, d=d)
        Ipred = tmp.predict(Imerge).compressed()

        # compute scaled intensities with a = b = d = 1, c = 0
        tmp.scale, tmp.offset = self.calc_scale_offset()
        I = tmp.I.compressed()
        sigma = tmp.sigma.compressed()

        # compute fitting matrices
        A = sparse.diags(Ipred / sigma) @ A0
        y = I / sigma
        AA = A.T @ A
        BBxy = Bxy.T @ Bxy
        BBz = Bz.T @ Bz
        Ay = A.T @ y
        yy = np.sum(y**2)

        return AA, BBxy, BBz, Ay, yy

    def fit(self, Imerge, alpha_xy_mult, alpha_z_mult, b=None, c=None, d=None):
        AA, BBxy, BBz, Ay, yy = self.calc_problem(Imerge, b=b, c=c, d=d)
        alpha_xy = alpha_xy_mult * AA.trace() / BBxy.trace()
        alpha_z = alpha_z_mult * AA.trace() / BBz.trace()
        G = AA + alpha_xy * BBxy + alpha_z * BBz
        if sparse.issparse(G):
            res = sparse.linalg.lsqr(G, Ay)
            ufit = res[0]
            note(lsqr_iters=int(res[2]), n_params=int(G.shape[0]), nnz_G=int(G.nnz))
        else:
            ufit = np.linalg.lstsq(G, Ay, rcond=None)[0]
        x2 = ufit @ AA @ ufit - 2 * ufit @ Ay + yy
        x2 = x2 / self.data.ih.count()
        self.model.set_u(ufit)
        return ufit, x2


class OffsetModelRefiner(_ModelRefiner):
    def add_model(self, dphi=2.5, nearest_degree=True, nphi=None, ns=31, ds=None):
        phi_points = self._calc_points(self.data.phi, dx=dphi, integer_limits=nearest_degree, nx=nphi)
        s_points = self._calc_points(self.data.s, dx=ds, integer_limits=False, nx=ns)
        self.model = OffsetModel(s_points, phi_points)
        return self

    def calc_value(self):
        if self.model is None:
            return 0.0
        return self.model.interp(self.data._s, self.data._phi)

    def calc_problem(self, Imerge, a=None, b=None, d=None):
        A0, Bx, By = self.model.interp_matrices(
            self.data.s.compressed(),
            self.data.phi.compressed(),
        )

        # compute predicted intensities a = 1, c = 0
        tmp = self.data.copy()
        tmp.scale, tmp.offset = self.calc_scale_offset(b=b)
        Ipred = tmp.predict(Imerge).compressed()

        # compute scaled intensities with b = 1 and c = 0
        tmp.scale, tmp.offset = self.calc_scale_offset(a=a, d=d)
        I = tmp.I.compressed()
        sigma = tmp.sigma.compressed()

        # compute fitting matrices
        A = sparse.diags(1 / sigma) @ A0
        y = (I - Ipred) / sigma
        AA = A.T @ A
        BBx = Bx.T @ Bx
        BBy = By.T @ By
        Ay = A.T @ y
        yy = np.sum(y**2)
        H = sparse.eye(A.shape[1])
        return AA, BBx, BBy, H, Ay, yy

    def fit(self, Imerge, alpha_x_mult, alpha_y_mult, alpha_c_mult, min_c, a=None, b=None, d=None):
        AA, BBx, BBy, H, Ay, yy = self.calc_problem(Imerge, a=a, b=b, d=d)
        alpha_x = alpha_x_mult * AA.trace() / BBx.trace()
        alpha_y = alpha_y_mult * AA.trace() / BBy.trace()
        # alpha_c = alpha_c_mult*AA.trace()/H.trace()
        alpha_c = alpha_c_mult * np.max(AA.diagonal()) / np.max(H.diagonal())
        G = AA + alpha_x * BBx + alpha_y * BBy + alpha_c * H
        if sparse.issparse(G):
            res = sparse.linalg.lsqr(G, Ay)
            ufit = res[0]
            note(lsqr_iters=int(res[2]), n_params=int(G.shape[0]), nnz_G=int(G.nnz))
        else:
            ufit = np.linalg.lstsq(G, Ay, rcond=None)[0]
        x2 = ufit @ AA @ ufit - 2 * ufit @ Ay + yy
        x2 = x2 / self.data.ih.count()
        if min_c is not None:
            ufit[ufit < min_c] = min_c  # apply threshold
            ufit += min_c - np.min(ufit)  # level shift

        return ufit, x2


class DetectorModelRefiner(_ModelRefiner):
    def add_model(self, nix=20, dix=None, niy=20, diy=None):
        ix_points = self._calc_points(self.data.ix, dx=dix, integer_limits=True, nx=nix)
        iy_points = self._calc_points(self.data.iy, dx=diy, integer_limits=True, nx=niy)
        self.model = DetectorModel(ix_points, iy_points)

    def calc_value(self):
        if self.model is None:
            return 1.0
        return self.model.interp(self.data._ix, self.data._iy)

    def calc_problem(self, Imerge, a=None, b=None, c=None):
        A0, B = self.model.interp_matrices(
            self.data.ix.compressed(),
            self.data.iy.compressed(),
        )

        # compute predicted intensities d = 1
        tmp = self.data.copy()
        tmp.scale, tmp.offset = self.calc_scale_offset(a=a, b=b, c=c)
        Ipred = tmp.predict(Imerge).compressed()

        # compute scaled intensities with a = b = d = 1, c = 0
        tmp.scale, tmp.offset = self.calc_scale_offset()
        I = tmp.I.compressed()
        sigma = tmp.sigma.compressed()

        # compute fitting matrices
        A = sparse.diags(Ipred / sigma) @ A0
        y = I / sigma
        AA = A.T @ A
        BB = B.T @ B
        Ay = A.T @ y
        yy = np.sum(y**2)

        return AA, BB, Ay, yy

    def fit(self, Imerge, alpha_mult, a=None, b=None, c=None):
        AA, BB, Ay, yy = self.calc_problem(Imerge, a=a, b=b, c=c)
        alpha = alpha_mult * AA.trace() / BB.trace()
        G = AA + alpha * BB
        if sparse.issparse(G):
            res = sparse.linalg.lsqr(G, Ay)
            ufit = res[0]
            note(lsqr_iters=int(res[2]), n_params=int(G.shape[0]), nnz_G=int(G.nnz))
        else:
            ufit = np.linalg.lstsq(G, rcond=None)[0]
        x2 = ufit @ AA @ ufit - 2 * ufit @ Ay + yy
        x2 = x2 / self.data.ih.count()
        return ufit, x2


class CombinedModelRefiner:
    def __init__(self, data, scaling_model=None, absorption_model=None, offset_model=None, detector_model=None):
        self.data = data
        self.scaling = ScalingModelRefiner(data, scaling_model)
        self.absorption = AbsorptionModelRefiner(data, absorption_model)
        self.offset = OffsetModelRefiner(data, offset_model)
        self.detector = DetectorModelRefiner(data, detector_model)

    @property
    def a(self):
        return self.absorption.value

    @property
    def b(self):
        return self.scaling.value

    @property
    def c(self):
        return self.offset.value

    @property
    def d(self):
        return self.detector.value

    def calc_scale_offset(self):
        scale = np.ones_like(self.data._I, dtype=np.double)
        offset = np.zeros_like(scale)
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        scale *= a * b * d
        offset += c / b
        return scale, offset

    def apply(self):
        """apply the scaling model to the data"""
        self.data.scale[:], self.data.offset[:] = self.calc_scale_offset()

    def bfit(self, *args, **kwargs):
        return self.scaling.refine(*args, a=self.a, c=self.c, d=self.d, **kwargs)

    def afit(self, *args, **kwargs):
        return self.absorption.refine(*args, b=self.b, c=self.c, d=self.d, **kwargs)

    def dfit(self, *args, **kwargs):
        return self.detector.refine(*args, a=self.a, b=self.b, c=self.c, **kwargs)

    def cfit(self, *args, **kwargs):
        return self.offset.refine(*args, a=self.a, b=self.b, d=self.d, **kwargs)


class BatchModelRefiner:
    def __init__(self, data, scaling_models=None, absorption_models=None, offset_models=None, detector_model=None):
        self.data = data
        self._batch_refiners = []

        if scaling_models is None:
            scaling_models = [None] * self.data.nbatches
        if absorption_models is None:
            absorption_models = [None] * self.data.nbatches
        if offset_models is None:
            offset_models = [None] * self.data.nbatches

        for ds, sm, am, om in zip(self.data.batches(), scaling_models, absorption_models, offset_models):
            self._batch_refiners.append(CombinedModelRefiner(ds, sm, am, om, detector_model))

        self.detector = DetectorModelRefiner(data, detector_model)

    @property
    def batch_refiners(self):
        """Access individual batch model refiners.

        Returns an iterator over CombinedModelRefiner instances, one per batch.
        This allows external code to iterate over refiners or access their models
        without directly coupling to the internal _batch_refiners implementation.

        Example:
            for refiner in batch_model_refiner.batch_refiners:
                model = refiner.scaling.model
        """
        return iter(self._batch_refiners)

    def add_scaling_models(self, *args, **kwargs):
        [br.scaling.add_model(*args, **kwargs) for br in self._batch_refiners]

    def add_offset_models(self, *args, **kwargs):
        [br.offset.add_model(*args, **kwargs) for br in self._batch_refiners]

    def add_absorption_models(self, *args, **kwargs):
        [br.absorption.add_model(*args, **kwargs) for br in self._batch_refiners]

    def add_detector_model(self, *args, **kwargs):
        tmp = DetectorModelRefiner(self.data)
        tmp.add_model(*args, **kwargs)
        self.detector = tmp
        for br in self._batch_refiners:
            br.detector.model = self.detector.model

    def bfit(self, *args, **kwargs):
        x2 = [br.bfit(*args, **kwargs) for br in self._batch_refiners]
        return np.mean(x2)

    def afit(self, *args, **kwargs):
        x2 = [br.afit(*args, **kwargs) for br in self._batch_refiners]
        return np.mean(x2)

    def cfit(self, *args, **kwargs):
        x2 = [br.cfit(*args, **kwargs) for br in self._batch_refiners]
        return np.mean(x2)

    @property
    def a(self):
        vals = []
        for br in self._batch_refiners:
            vals.append(br.a * np.ones_like(br.data._I, dtype=np.double))
        return np.concatenate(vals)

    @property
    def b(self):
        vals = []
        for br in self._batch_refiners:
            vals.append(br.b * np.ones_like(br.data._I, dtype=np.double))
        return np.concatenate(vals)

    @property
    def c(self):
        vals = []
        for br in self._batch_refiners:
            vals.append(br.c * np.ones_like(br.data._I, dtype=np.double))
        return np.concatenate(vals)

    def dfit(self, *args, **kwargs):
        for br in self._batch_refiners:
            br.detector._value = None  # trigger re-compute on next apply
        return self.detector.refine(*args, a=self.a, b=self.b, c=self.c, **kwargs)

    def apply(self):
        [br.apply() for br in self._batch_refiners]
