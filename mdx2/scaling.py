"""Scaling models and the scaling engine used by mdx2.scale.

Models (ScalingModel b(phi), AbsorptionModel a(ix, iy, phi), OffsetModel c(s, phi), DetectorModel d(ix, iy)) are
piecewise-linear on uniform control-point grids (InterpLin1/2/3) and are what scale files store. ScaledData and
BatchModelRefiner implement mdx2's alternating least squares: models, grids (from the unmasked data), penalties, the
lambda recipe alpha tr(A'A)/tr(B'B), the lsqr solve, chi2, the offset threshold / level shift and one-sided outlier
masking. The passes over the observations are compiled multi-threaded kernels (mdx2._scaling_kernels): A'A, A'y and
y'y for a fit are accumulated in ONE pass with each model's weights and targets computed inline, A'A as neighbour
slots on the control-point grid; merging and masking are threaded; masked observations get zero weight. Threads:
mdx2.parallel.set_threads. The previous implementation is mdx2.scaling_reference.
"""

import copy

import numba as nb
import numpy as np
from nexusformat.nexus import NXdata, NXfield
from scipy import sparse
from scipy.sparse.linalg import lsqr
from threadpoolctl import threadpool_limits

from mdx2._scaling_kernels import (
    k_apply,
    k_group_order,
    k_merge,
    k_normal,
    k_outliers,
    k_split_half,
    k_values,
)
from mdx2.parallel import check_count
from mdx2.profiling import note

try:
    import xarray as xr
except ImportError:
    pass  # fail silently... xarray features still in development


class InterpLin3:
    def __init__(self, x, y, z, wx, wy, wz):
        self.x = x  # points to interpolate at in the x direction
        self.y = y  # points to interpolate at in the y direction
        self.z = z  # points to interpolate at in the z direction
        self.wx = wx  # the control points in the x direction, increasing
        self.wy = wy  # the control points in the y direction, increasing
        self.wz = wz  # the control points in the z direction, increasing

    @property
    def shape(self):
        return self.wx.size, self.wy.size, self.wz.size

    def _map(self):
        Nx, Ny, Nz = self.shape
        isinside = (
            (self.x >= self.wx[0])
            & (self.x <= self.wx[-1])
            & (self.y >= self.wy[0])
            & (self.y <= self.wy[-1])
            & (self.z >= self.wz[0])
            & (self.z <= self.wz[-1])
        )
        ind = np.nonzero(isinside)[0]
        x = self.x[ind]
        y = self.y[ind]
        z = self.z[ind]
        fx = np.interp(x, self.wx, np.arange(Nx))
        fy = np.interp(y, self.wy, np.arange(Ny))
        fz = np.interp(z, self.wz, np.arange(Nz))

        xbin = fx.astype(int)
        ybin = fy.astype(int)
        zbin = fz.astype(int)
        xbin[xbin == (Nx - 1)] -= 1  # fix edge cases
        ybin[ybin == (Ny - 1)] -= 1
        zbin[zbin == (Nz - 1)] -= 1
        xfrac = fx - xbin
        yfrac = fy - ybin
        zfrac = fz - zbin
        return ind, xbin, ybin, zbin, xfrac, yfrac, zfrac

    def _sub2ind(self, ix, iy, iz, Nx=None, Ny=None, Nz=None):
        if Nx is None:
            Nx = self.wx.size
        if Ny is None:
            Ny = self.wy.size
        if Nz is None:
            Nz = self.wz.size
        # ind = ix + Nx*(iy + Ny*iz)
        ind = np.ravel_multi_index((ix, iy, iz), (Nx, Ny, Nz), order="F")
        return ind

    def interp(self, p):
        ind, xbin, ybin, zbin, xfrac, yfrac, zfrac = self._map()

        vals = np.zeros_like(self.x)
        p = p.flatten(order="F")
        vals[ind] = (
            p[self._sub2ind(xbin, ybin, zbin)] * (1 - zfrac) * (1 - xfrac) * (1 - yfrac)
            + p[self._sub2ind(xbin + 1, ybin, zbin)] * (1 - zfrac) * xfrac * (1 - yfrac)
            + p[self._sub2ind(xbin, ybin + 1, zbin)] * (1 - zfrac) * (1 - xfrac) * yfrac
            + p[self._sub2ind(xbin + 1, ybin + 1, zbin)] * (1 - zfrac) * xfrac * yfrac
            + p[self._sub2ind(xbin, ybin, zbin + 1)] * zfrac * (1 - xfrac) * (1 - yfrac)
            + p[self._sub2ind(xbin + 1, ybin, zbin + 1)] * zfrac * xfrac * (1 - yfrac)
            + p[self._sub2ind(xbin, ybin + 1, zbin + 1)] * zfrac * (1 - xfrac) * yfrac
            + p[self._sub2ind(xbin + 1, ybin + 1, zbin + 1)] * zfrac * xfrac * yfrac
        )

        return vals

    @property
    def A(self):
        ind, xbin, ybin, zbin, xfrac, yfrac, zfrac = self._map()
        vals = np.concatenate(
            (
                (1 - zfrac) * (1 - xfrac) * (1 - yfrac),
                (1 - zfrac) * xfrac * (1 - yfrac),
                (1 - zfrac) * (1 - xfrac) * yfrac,
                (1 - zfrac) * xfrac * yfrac,
                zfrac * (1 - xfrac) * (1 - yfrac),
                zfrac * xfrac * (1 - yfrac),
                zfrac * (1 - xfrac) * yfrac,
                zfrac * xfrac * yfrac,
            )
        )
        row_index = np.concatenate((ind, ind, ind, ind, ind, ind, ind, ind))
        x_index = np.concatenate((xbin, xbin + 1, xbin, xbin + 1, xbin, xbin + 1, xbin, xbin + 1))
        y_index = np.concatenate((ybin, ybin, ybin + 1, ybin + 1, ybin, ybin, ybin + 1, ybin + 1))
        z_index = np.concatenate((zbin, zbin, zbin, zbin, zbin + 1, zbin + 1, zbin + 1, zbin + 1))
        col_index = self._sub2ind(x_index, y_index, z_index)
        shape = (self.x.size, self.wx.size * self.wy.size * self.wz.size)
        return sparse.coo_matrix((vals, (row_index, col_index)), shape=shape)

    @property
    def Bx(self):
        raise NotImplementedError

    @property
    def By(self):
        raise NotImplementedError

    @property
    def Bz(self):
        Nx, Ny, Nz = self.shape
        k, l, m = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
        ind_left = self._sub2ind(k[:, :, :-2], l[:, :, :-2], m[:, :, :-2]).ravel(order="F")
        ind_self = self._sub2ind(k[:, :, 1:-1], l[:, :, 1:-1], m[:, :, 1:-1]).ravel(order="F")
        ind_right = self._sub2ind(k[:, :, 2:], l[:, :, 2:], m[:, :, 2:]).ravel(order="F")
        ind_rows = self._sub2ind(k[:, :, :-2], l[:, :, :-2], m[:, :, :-2]).ravel(order="F")
        nrows = Nx * Ny * (Nz - 2)
        row_index = np.concatenate((ind_rows, ind_rows, ind_rows))
        col_index = np.concatenate((ind_left, ind_self, ind_right))
        vals = np.concatenate((np.full(nrows, -0.5), np.full(nrows, 1.0), np.full(nrows, -0.5)))
        shape = (nrows, Nx * Ny * Nz)
        return sparse.coo_matrix((vals, (row_index, col_index)), shape=shape)

    @property
    def B(self):
        Nx, Ny, Nz = self.shape
        k, l, m = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")

        # Neighbor indices
        right = self._sub2ind(k[1:, :, :], l[1:, :, :], m[1:, :, :]).ravel(order="F")
        left = self._sub2ind(k[:-1, :, :], l[:-1, :, :], m[:-1, :, :]).ravel(order="F")
        up = self._sub2ind(k[:, 1:, :], l[:, 1:, :], m[:, 1:, :]).ravel(order="F")
        down = self._sub2ind(k[:, :-1, :], l[:, :-1, :], m[:, :-1, :]).ravel(order="F")
        front = self._sub2ind(k[:, :, :-1], l[:, :, :-1], m[:, :, :-1]).ravel(order="F")
        back = self._sub2ind(k[:, :, 1:], l[:, :, 1:], m[:, :, 1:]).ravel(order="F")

        # Flatten all neighbor indices
        row_index = np.concatenate((left, right, down, up, front, back))
        col_index = np.concatenate((right, left, up, down, back, front))
        vals = np.ones_like(row_index, dtype=np.double)
        shape = (Nx * Ny * Nz, Nx * Ny * Nz)
        L = sparse.coo_matrix((vals, (row_index, col_index)), shape=shape)

        # Weight according to number of neighbors
        # NOTE: this may fail if Nx=Ny=Nz=1
        nn = np.asarray(L.sum(axis=1)).flatten()
        W = sparse.diags(1 / nn)
        L = W @ L

        # Center minus average of neighbors
        return sparse.eye(shape[0]) - L

    @property
    def Bxy(self):
        Nx, Ny, Nz = self.shape
        k, l, m = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
        ind_left = self._sub2ind(k[:-1, :, :], l[:-1, :, :], m[:-1, :, :]).ravel(order="F")
        ind_right = self._sub2ind(k[1:, :, :], l[1:, :, :], m[1:, :, :]).ravel(order="F")
        ind_up = self._sub2ind(k[:, :-1, :], l[:, :-1, :], m[:, :-1, :]).ravel(order="F")
        ind_down = self._sub2ind(k[:, 1:, :], l[:, 1:, :], m[:, 1:, :]).ravel(order="F")
        row_index = np.concatenate((ind_left, ind_right, ind_up, ind_down))
        col_index = np.concatenate((ind_right, ind_left, ind_down, ind_up))
        vals = np.ones_like(row_index, dtype=np.double)
        shape = (Nx * Ny * Nz, Nx * Ny * Nz)
        L = sparse.coo_matrix((vals, (row_index, col_index)), shape=shape)
        # weight accordint to the number of neighbors
        nn = np.asarray(L.sum(axis=1)).flatten()
        return sparse.eye(shape[0]) - sparse.diags(1 / nn) @ L


class InterpLin2:
    def __init__(self, x, y, wx, wy):
        self.x = x  # points to interpolate at in the x direction
        self.y = y  # points to interpolate at in the y direction
        self.wx = wx  # the control points in the x direction, increasing
        self.wy = wy  # the control points in the y direction, increasing

    @property
    def shape(self):
        return self.wx.size, self.wy.size

    def _map(self):
        Nx, Ny = self.shape
        fx = np.interp(self.x, self.wx, np.arange(Nx), left=np.nan, right=np.nan)
        fy = np.interp(self.y, self.wy, np.arange(Ny), left=np.nan, right=np.nan)
        ind = np.nonzero(~np.isnan(fx) & ~np.isnan(fy))[0].astype(int)
        fx = fx[ind]
        fy = fy[ind]
        xbin = np.floor(fx).astype(int)
        ybin = np.floor(fy).astype(int)
        xbin[xbin == (Nx - 1)] -= 1  # fix edge cases
        ybin[ybin == (Ny - 1)] -= 1
        xfrac = fx - xbin
        yfrac = fy - ybin
        return ind, xbin, ybin, xfrac, yfrac

    def _sub2ind(self, ix, iy, Nx=None, Ny=None):
        """convert 2D array index to linear index"""
        if Nx is None:
            Nx = self.wx.size
        if Ny is None:
            Ny = self.wy.size
        # ind = ix + Nx*iy;
        ind = np.ravel_multi_index((ix, iy), (Nx, Ny), order="F")
        return ind

    def interp(self, p):
        ind, xbin, ybin, xfrac, yfrac = self._map()
        vals = np.zeros_like(self.x)
        p = p.flatten(order="F")
        vals[ind] = (
            p[self._sub2ind(xbin, ybin)] * (1 - xfrac) * (1 - yfrac)
            + p[self._sub2ind(xbin + 1, ybin)] * xfrac * (1 - yfrac)
            + p[self._sub2ind(xbin, ybin + 1)] * (1 - xfrac) * yfrac
            + p[self._sub2ind(xbin + 1, ybin + 1)] * xfrac * yfrac
        )
        return vals

    @property
    def A(self):
        ind, xbin, ybin, xfrac, yfrac = self._map()
        vals = np.concatenate(((1 - xfrac) * (1 - yfrac), xfrac * (1 - yfrac), (1 - xfrac) * yfrac, xfrac * yfrac))
        row_index = np.concatenate((ind, ind, ind, ind))
        x_index = np.concatenate((xbin, xbin + 1, xbin, xbin + 1))
        y_index = np.concatenate((ybin, ybin, ybin + 1, ybin + 1))
        col_index = self._sub2ind(x_index, y_index)
        shape = (self.x.size, self.wx.size * self.wy.size)
        return sparse.coo_matrix((vals, (row_index, col_index)), shape=shape)

    @property
    def Bx(self):
        Nx, Ny = self.shape
        k, l = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
        ind_left = self._sub2ind(k[:-2, :], l[:-2, :]).ravel(order="F")
        ind_self = self._sub2ind(k[1:-1, :], l[1:-1, :]).ravel(order="F")
        ind_right = self._sub2ind(k[2:, :], l[2:, :]).ravel(order="F")
        ind_rows = self._sub2ind(k[:-2, :], l[:-2, :], Nx=Nx - 2).ravel(order="F")
        nrows = (Nx - 2) * Ny
        row_index = np.concatenate((ind_rows, ind_rows, ind_rows))
        col_index = np.concatenate((ind_left, ind_self, ind_right))
        vals = np.concatenate((np.full(nrows, -0.5), np.full(nrows, 1.0), np.full(nrows, -0.5)))
        shape = (nrows, Nx * Ny)
        return sparse.coo_matrix((vals, (row_index, col_index)), shape=shape)

    @property
    def By(self):
        Nx, Ny = self.shape
        k, l = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
        ind_left = self._sub2ind(k[:, :-2], l[:, :-2]).ravel(order="F")
        ind_self = self._sub2ind(k[:, 1:-1], l[:, 1:-1]).ravel(order="F")
        ind_right = self._sub2ind(k[:, 2:], l[:, 2:]).ravel(order="F")
        ind_rows = self._sub2ind(k[:, :-2], l[:, :-2]).ravel(order="F")
        nrows = Nx * (Ny - 2)
        row_index = np.concatenate((ind_rows, ind_rows, ind_rows))
        col_index = np.concatenate((ind_left, ind_self, ind_right))
        vals = np.concatenate((np.full(nrows, -0.5), np.full(nrows, 1.0), np.full(nrows, -0.5)))
        shape = (nrows, Nx * Ny)
        return sparse.coo_matrix((vals, (row_index, col_index)), shape=shape)

    @property
    def B(self):
        Nx, Ny = self.shape
        k, l = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
        ind_left = self._sub2ind(k[:-1, :], l[:-1, :]).ravel(order="F")
        ind_right = self._sub2ind(k[1:, :], l[1:, :]).ravel(order="F")
        ind_up = self._sub2ind(k[:, :-1], l[:, :-1]).ravel(order="F")
        ind_down = self._sub2ind(k[:, 1:], l[:, 1:]).ravel(order="F")
        row_index = np.concatenate((ind_left, ind_right, ind_up, ind_down))
        col_index = np.concatenate((ind_right, ind_left, ind_down, ind_up))
        vals = np.ones_like(row_index, dtype=np.double)
        shape = (Nx * Ny, Nx * Ny)
        L = sparse.coo_matrix((vals, (row_index, col_index)), shape=shape)
        # weight accordint to the number of neighbors
        nn = np.asarray(L.sum(axis=1)).flatten()
        return sparse.eye(shape[0]) - sparse.diags(1 / nn) @ L


class InterpLin1:
    def __init__(self, x, w):
        self.x = x  # the points to interpolate on
        self.w = w  # the control points, increasing

    @property
    def shape(self):
        return self.w.size

    def _map(self):
        N = self.shape
        f = np.interp(self.x, self.w, np.arange(N), left=np.nan, right=np.nan)
        x_index = np.nonzero(~np.isnan(f))[0].astype(int)
        f = f[x_index]
        bin_index = np.floor(f).astype(int)
        bin_index[bin_index == (N - 1)] -= 1  # fix edge cases
        bin_index[bin_index == self.w.size - 1] -= 1  # deal with edge case
        bin_fraction = f - bin_index
        return x_index, bin_index, bin_fraction

    def interp(self, p):
        ix, bin_ix, frac = self._map()
        vals = np.zeros_like(self.x)
        vals[ix] = p[bin_ix] * (1 - frac) + p[bin_ix + 1] * frac
        return vals

    @property
    def A(self):
        ix, bin_ix, frac = self._map()
        vals = np.concatenate((1 - frac, frac))
        row_index = np.concatenate((ix, ix))
        col_index = np.concatenate((bin_ix, bin_ix + 1))
        shape = (self.x.size, self.w.size)
        return sparse.coo_matrix((vals, (row_index, col_index)), shape=shape)

    @property
    def B(self):
        N = self.shape
        row_index = np.concatenate((np.arange(N - 2), np.arange(N - 2), np.arange(N - 2)))
        col_index = np.concatenate((np.arange(N - 2), np.arange(1, N - 1), np.arange(2, N)))
        vals = np.concatenate((np.full(N - 2, -0.5), np.full(N - 2, 1), np.full(N - 2, -0.5)))
        return sparse.coo_matrix((vals, (row_index, col_index)), shape=(N - 2, N))


class _BaseModel:
    """Common methods for scaling models"""

    # properties:
    #    x,y,... = coordinates to interpolate at
    #    wx,wy,... = axes of control point grid
    #    u = value at control points
    #
    # for compatibility with MATLAB code, use Fortran array order
    # u.shape == (Nx,Ny,...)
    @property
    def shape(self):
        pass  # implemented in subclass

    def interp_matrices(self, *args):
        pass  # implemented in subclass

    def interp(self, x):
        pass  # implemented in subclass

    def to_nexus(self):
        pass  # implemented in subclass

    @staticmethod
    def from_nexus(n):
        pass  # implemented in subclass

    def to_xarray(self):
        pass  # implemented in subclass

    @staticmethod
    def from_xarray(da):
        pass  # implemented in subclass

    def set_u(self, val):
        self.u = val.reshape(self.shape, order="F")

    def plot(self, *args, **kwargs):
        self.to_xarray.plot(*args, **kwargs)

    def copy(self):
        SM = copy.copy(self)
        SM.u = copy.copy(SM.u)
        return SM


class ScalingModel(_BaseModel):
    """A scale factor the varies only along one coordinate"""

    def __init__(self, phi, u=None):
        self.x = phi
        self.u = u

    @property
    def shape(self):
        return self.x.size

    def interp_matrices(self, x):
        Lin1 = InterpLin1(x, self.x)
        return Lin1.A, Lin1.B

    def interp(self, x):
        if self.u is None:
            val = np.ones_like(x, dtype=np.double)
        else:
            val = InterpLin1(x, self.x).interp(self.u)
        return val

    def to_nexus(self):
        phi = NXfield(self.x, name="phi")
        u = NXfield(self.u, name="u")
        return NXdata(signal=u, axes=[phi], name="scaling_model")

    @staticmethod
    def from_nexus(nxdata):
        return ScalingModel(nxdata.phi.nxvalue, nxdata.u.nxvalue)

    def to_xarray(self):
        da = xr.DataArray(self.u, dims=("phi"), coords={"phi": self.x})
        return da

    @staticmethod
    def from_xarray(da):
        x = da.coords["phi"].data
        return ScalingModel(x, da.data)


class AbsorptionModel(_BaseModel):
    """A smooth scale factor the varies along 3 coordinates"""

    def __init__(self, ix, iy, phi, u=None):
        self.x = ix
        self.y = iy
        self.z = phi
        self.u = u

    @property
    def shape(self):
        return self.x.size, self.y.size, self.z.size

    def interp_matrices(self, x, y, z):
        Lin3 = InterpLin3(x, y, z, self.x, self.y, self.z)
        return Lin3.A, Lin3.Bxy, Lin3.Bz

    def interp(self, x, y, z):
        if self.u is None:
            val = np.ones_like(x, dtype=np.double)
        else:
            val = InterpLin3(x, y, z, self.x, self.y, self.z).interp(self.u)
        return val

    def to_nexus(self):
        ix = NXfield(self.x, name="ix")
        iy = NXfield(self.y, name="iy")
        phi = NXfield(self.z, name="phi")
        u = NXfield(self.u, name="u")
        return NXdata(signal=u, axes=[ix, iy, phi], name="absorption_model")

    @staticmethod
    def from_nexus(nxdata):
        return AbsorptionModel(nxdata.ix.nxvalue, nxdata.iy.nxvalue, nxdata.phi.nxvalue, nxdata.u.nxvalue)

    def to_xarray(self):
        da = xr.DataArray(self.u, dims=("ix", "iy", "phi"), coords={"ix": self.x, "iy": self.x, "phi": self.z})
        return da

    @staticmethod
    def from_xarray(da):
        x = da.coords["ix"].data
        y = da.coords["iy"].data
        z = da.coords["phi"].data
        return AbsorptionModel(x, y, z, da.data)


class OffsetModel(_BaseModel):
    """A smooth offset the varies along 2 coordinates"""

    def __init__(self, s, phi, u=None):
        self.x = s
        self.y = phi
        self.u = u

    @property
    def shape(self):
        return self.x.size, self.y.size

    def interp_matrices(self, x, y):
        Lin2 = InterpLin2(x, y, self.x, self.y)
        return Lin2.A, Lin2.Bx, Lin2.By

    def interp(self, x, y):
        if self.u is None:
            val = np.zeros_like(x, dtype=np.double)
        else:
            val = InterpLin2(x, y, self.x, self.y).interp(self.u)
        return val

    def to_nexus(self):
        s = NXfield(self.x, name="s")
        phi = NXfield(self.y, name="phi")
        u = NXfield(self.u, name="u")
        return NXdata(signal=u, axes=[s, phi], name="offset_model")

    @staticmethod
    def from_nexus(nxdata):
        return OffsetModel(nxdata.s.nxvalue, nxdata.phi.nxvalue, nxdata.u.nxvalue)

    def to_xarray(self):
        da = xr.DataArray(self.u, dims=("s", "phi"), coords={"s": self.x, "phi": self.y})
        return da

    @staticmethod
    def from_xarray(da):
        x = da.coords["s"].data
        y = da.coords["phi"].data
        return OffsetModel(x, y, da.data)


class DetectorModel(_BaseModel):
    """A smooth scale factor the varies along the detector face"""

    def __init__(self, ix, iy, u=None):
        self.x = ix
        self.y = iy
        self.u = u

    @property
    def shape(self):
        return self.x.size, self.y.size

    def interp_matrices(self, x, y):
        Lin3 = InterpLin2(x, y, self.x, self.y)
        return Lin3.A, Lin3.B

    def interp(self, x, y):
        if self.u is None:
            val = np.ones_like(x, dtype=np.double)
        else:
            val = InterpLin2(x, y, self.x, self.y).interp(self.u)
        return val

    def to_nexus(self):
        ix = NXfield(self.x, name="ix")
        iy = NXfield(self.y, name="iy")
        u = NXfield(self.u, name="u")
        return NXdata(signal=u, axes=[ix, iy], name="detector_model")

    @staticmethod
    def from_nexus(nxdata):
        return DetectorModel(nxdata.ix.nxvalue, nxdata.iy.nxvalue, nxdata.u.nxvalue)

    def to_xarray(self):
        da = xr.DataArray(self.u, dims=("ix", "iy"), coords={"ix": self.x, "iy": self.x})
        return da

    @staticmethod
    def from_xarray(da):
        x = da.coords["ix"].data
        y = da.coords["iy"].data
        return AbsorptionModel(x, y, da.data)


KIND_B, KIND_A, KIND_C, KIND_D = 0, 1, 2, 3


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


class ScaledData:
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
        self._order, self._gstart = k_group_order(self._ih, self._ihmax + 1)

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

    def merge(self, mask=None):
        """Inverse-variance weighted mean per group of the scaled observations not masked (by `mask`, default
        self.mask). Returns Im, sigma_m (NaN for empty groups) and counts."""
        mask = self.mask if mask is None else np.ascontiguousarray(mask, dtype=np.bool_)
        G = self._ihmax + 1
        Im, sigm, counts = np.empty(G), np.empty(G), np.empty(G, dtype=np.int64)
        k_merge(self._gstart, self._order, mask, self._I, self._sigma, self.scale, self.offset, Im, sigm, counts)
        check_count("mdx2.scaling merge", int(counts.sum()), int(np.count_nonzero(~mask)))
        return Im, sigm, counts

    def weights(self):
        """Inverse variances of the scaled observations, (scale / sigma)^2; 0 where masked."""
        with np.errstate(divide="ignore", invalid="ignore"):
            w = (self.scale / self._sigma) ** 2
        w[self.mask] = 0.0
        return w

    def split_half(self, w, jitter, flips):
        """mdx2.merge's weighted random half split: within each group, observations in random order (by `jitter`)
        go to the first half once their cumulative weight `w` passes half the group's total, swapped where
        `flips[group]` is set. Returns a boolean array (True = first half)."""
        out = np.empty(len(self), dtype=np.bool_)
        k_split_half(
            self._gstart,
            self._order,
            np.ascontiguousarray(jitter, dtype=np.float64),
            np.ascontiguousarray(w, dtype=np.float64),
            np.ascontiguousarray(flips, dtype=np.int64),
            out,
        )
        return out

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


class BatchModelRefiner:
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
        views = []
        for terms in self.terms:
            models = {k: t.model for k, t in terms.items()}
            if self.dterm is not None:
                models["d"] = self.dterm.model
            views.append(_BatchView(models))
        return iter(views)

    _COORDS = dict(b=("_phi",), a=("_ix", "_iy", "_phi"), c=("_s", "_phi"), d=("_ix", "_iy"))

    def set_model(self, k, name, model):
        """Use a given model (e.g. from a scale file) on batch k: name "b" (scaling), "a" (absorption), "c" (offset)
        or "d" (detector, per batch as in mdx2.merge). Values update at once; apply() then sets scale and offset."""
        D = self.data
        t = _Term(model, tuple(getattr(D, a) for a in self._COORDS[name]), D._batches[k], [])
        self.terms[k][name] = t
        self._update_values(name, t)

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
        check_count("mdx2.scaling fit", int(count), D.count(t.rows))
        AA = g.to_csr(slots)
        trAA = AA.diagonal().sum()
        G = AA.copy()
        for a_mult, BB in zip(alphas, t.BB):
            G = G + (a_mult * trAA / BB.diagonal().sum()) * BB
        if ridge is not None:  # mdx2 offset: alpha_c * max(diag(A'A)) * I
            G = G + ridge * np.max(AA.diagonal()) * sparse.eye(g.P)
        with threadpool_limits(limits=1, user_api="blas"):  # lsqr's vector operations are tiny: no BLAS threads
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
