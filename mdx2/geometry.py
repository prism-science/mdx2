import numba as nb
import numexpr as ne
import numpy as np
from nexusformat.nexus import NXdata, NXfield, NXgroup, NXsample

from mdx2.dxtbx_machinery import Experiment
from mdx2.io import array_field, loadobj, nxload
from mdx2.utils import interp2, interp3, interp_g2g_trilinear


class GaussianPeak:
    """Handy functions for 3d Gaussians"""

    def __init__(self, r0, sigma):
        self.r0 = r0
        self.sigma = sigma

    def __str__(self):
        return f"r0 = {self.r0}\nsigma =\n{self.sigma}"

    @staticmethod
    def fit_to_points(x, y, z, sigma_cutoff=1):
        """fit quadratic form to a set of points"""
        r = np.stack((x, y, z)).T
        r0 = np.mean(r, axis=0)
        u, s, vh = np.linalg.svd(r - r0, full_matrices=False)
        v = vh.T
        sigma = v @ np.diag(s / np.sqrt(u.shape[0]))

        GP = GaussianPeak(r0, sigma)

        if sigma_cutoff is None:
            return GP
        else:
            d2 = GP.quadform_at_points(x, y, z)
            is_outlier = d2 > sigma_cutoff**2
            x = x[~is_outlier]
            y = y[~is_outlier]
            z = z[~is_outlier]
            return GaussianPeak.fit_to_points(x, y, z, sigma_cutoff=None), is_outlier

    def quadform_at_points(self, x, y, z):
        r = np.stack((x, y, z)).reshape(3, -1)
        d = np.linalg.inv(self.sigma) @ (r - self.r0[:, np.newaxis])
        return np.sum(d * d, axis=0).reshape(x.shape)

    def mask(self, x, y, z, sigma_cutoff=1):
        # assume x,y,z are all the same shape
        d2 = self.quadform_at_points(x, y, z)
        return d2 < sigma_cutoff**2

    @staticmethod
    def from_nexus(gp):
        return GaussianPeak(np.array(gp.r0), np.array(gp.sigma))

    def to_nexus(self):
        return NXgroup(name="gaussian_peak", sigma=self.sigma, r0=self.r0)


class GridData:
    def __init__(self, axes, data, axes_names=None):
        self.axes = axes  # tuple of length 3 with axes of length l,m,n
        self.data = data  # stack of arrays of equal size: l by m by n by K
        if axes_names is None:
            axes_names = [f"axis{j}" for j in range(len(axes))]
        self.axes_names = axes_names

    def regrid(self, new_axes):
        new_data = interp_g2g_trilinear(*self.axes, self.data, *new_axes)
        return GridData(new_axes, new_data)

    @property
    def ndims(self):
        return len(self.axes)

    @property
    def ncols(self):
        if len(self.data.shape) == self.ndims:
            return 1
        else:
            return self.data.shape[self.ndims]

    @property
    def data_unstacked(self):
        if self.ncols == 1:  # single 3D array
            return [self.data]
        else:
            return [self.data[..., ind] for ind in range(self.ncols)]

    def split(self):
        return [GridData(self.axes, data) for data in self.data_unstacked]

    def interpolate(self, *args, **kwargs):
        outdata = []
        for col in self.data_unstacked:
            if self.ndims == 3:
                vals = interp3(*self.axes, col, *args, **kwargs)
            elif self.ndims == 2:
                vals = interp2(*self.axes, col, *args, **kwargs)
            else:
                raise (RuntimeError("interpolation not implemented yet for dimensions other and 2 or 3"))
            outdata.append(vals)
        if len(outdata) == 1:
            return outdata[0]
        else:
            return np.stack(outdata, axis=len(args[0].shape))

    def to_nexus(self):
        axes = [NXfield(ax, name=n) for ax, n in zip(self.axes, self.axes_names)]
        return NXdata(array_field(self.data, "signal"), axes, name="grid_data")

    @staticmethod
    def from_nexus(nxdata):
        data_name = nxdata.attrs["signal"]
        data = nxdata[data_name].nxvalue
        axes_names = nxdata.attrs["axes"]
        axes = [nxdata[name].nxvalue for name in axes_names]
        return GridData(axes, data, axes_names=axes_names)


class MillerIndex(GridData):
    """array of Miller indices"""

    # This is a container class for computed geometric corrections, mainly to handle file format conversion

    def __init__(self, phi, iy, ix, h, k, l):  # noqa: E741
        self.phi = phi
        self.iy = iy
        self.ix = ix
        self.h = h
        self.k = k
        self.l = l

    @property
    def data(self):
        return np.stack((self.h, self.k, self.l), axis=3)

    @property
    def axes(self):
        return (self.phi, self.iy, self.ix)

    def regrid(self, new_phi, new_iy, new_ix):
        new_axes = (new_phi, new_iy, new_ix)
        G3D = GridData.regrid(self, new_axes)
        hkl = tuple(G3D.data_unstacked)
        return MillerIndex(*new_axes, *hkl)

    def interpolate(self, phi, iy, ix, **kwargs):
        # each component directly: the data property stacks h, k, l into a new array on every access, and
        # GridData.interpolate reads it several times (Eiger 16M grid: ~10 GB of copies for 3.2 GB of indices).
        # Same interp3 on the same values, so the result is identical.
        return tuple(interp3(self.phi, self.iy, self.ix, a, phi, iy, ix, **kwargs) for a in (self.h, self.k, self.l))

    @staticmethod
    def from_expt(exptfile, sample_spacing=[1, 10, 10]):
        # sample spacing (phi, y, x) in units of (degrees, pixels, pixels)
        E = Experiment.from_file(exptfile)
        phi, iy, ix = E.calc_scan_axes(centered=False, spacing=sample_spacing)
        h, k, l = E.calc_hkl_on_grid(phi, iy, ix)
        return MillerIndex(phi, iy, ix, h, k, l)

    def to_nexus(self):
        ix = NXfield(self.ix, name="ix")
        iy = NXfield(self.iy, name="iy")
        phi = NXfield(self.phi, name="phi", units="degree")
        axes = [phi, iy, ix]
        nxgroup = NXgroup(
            name="miller_index",
            h=NXdata(array_field(self.h, "signal"), axes),
            k=NXdata(array_field(self.k, "signal"), axes),
            l=NXdata(array_field(self.l, "signal"), axes),
        )
        return nxgroup

    @staticmethod
    def from_nexus(nxgroup):
        def get_value(nxdata):
            return nxdata[nxdata.attrs["signal"]].nxvalue

        phi = nxgroup.h.phi.nxvalue
        iy = nxgroup.h.iy.nxvalue
        ix = nxgroup.h.ix.nxvalue

        return MillerIndex(
            phi,
            iy,
            ix,
            get_value(nxgroup.h),
            get_value(nxgroup.k),
            get_value(nxgroup.l),
        )


class Corrections(GridData):
    """geometric corrections"""

    # This is a container class for computed geometric corrections, mainly to handle file format conversion

    def __init__(
        self,
        iy,
        ix,
        solid_angle=None,
        efficiency=None,
        attenuation=None,
        polarization=None,
        inverse_lorentz=None,
        d3s=None,
    ):
        self.iy = iy
        self.ix = ix
        self.solid_angle = solid_angle
        self.efficiency = efficiency
        self.attenuation = attenuation
        self.polarization = polarization
        self.inverse_lorentz = inverse_lorentz
        self.d3s = d3s

    _data_keys = ["solid_angle", "efficiency", "attenuation", "polarization", "inverse_lorentz", "d3s"]

    @property
    def data(self):
        all_data = [self.__dict__[key] for key in self._data_keys]
        return np.stack(all_data, axis=2)

    @property
    def axes(self):
        return (self.iy, self.ix)

    def interpolate(self, iy, ix, order=1):
        outdata = GridData.interpolate(self, iy, ix, order=order)
        kvpairs = {key: outdata[..., ind] for ind, key in enumerate(self._data_keys)}
        return kvpairs

    def regrid(self, new_iy, new_ix):
        new_axes = (new_iy, new_ix)
        G = GridData.regrid(self, new_axes)
        kvpairs = {key: val for key, val in zip(self._data_keys, G.data_unstacked)}
        return Corrections(*G.axes, **kvpairs)

    @staticmethod
    def from_expt(exptfile, sample_spacing=[10, 10]):
        # sample spacing (y, x) in units of pixels
        E = Experiment.from_file(exptfile)
        _, iy, ix = E.calc_scan_axes(centered=False, spacing=(1,) + tuple(sample_spacing))
        corr = E.calc_corrections_on_grid(iy, ix)
        return Corrections(iy, ix, **corr)

    def to_nexus(self):
        ix = NXfield(self.ix, name="ix")
        iy = NXfield(self.iy, name="iy")
        axes = [iy, ix]
        nxgroup = NXgroup(
            name="corrections",
            solid_angle=NXdata(array_field(self.solid_angle, "signal"), axes),
            efficiency=NXdata(array_field(self.efficiency, "signal"), axes),
            attenuation=NXdata(array_field(self.attenuation, "signal"), axes),
            polarization=NXdata(array_field(self.polarization, "signal"), axes),
            inverse_lorentz=NXdata(array_field(self.inverse_lorentz, "signal"), axes),
            d3s=NXdata(array_field(self.d3s, "signal"), axes),
        )
        return nxgroup

    @staticmethod
    def from_nexus(nxgroup):
        def get_value(nxdata):
            return nxdata[nxdata.attrs["signal"]].nxvalue

        iy = nxgroup.solid_angle.iy.nxvalue
        ix = nxgroup.solid_angle.ix.nxvalue
        return Corrections(
            iy,
            ix,
            solid_angle=get_value(nxgroup.solid_angle),
            efficiency=get_value(nxgroup.efficiency),
            attenuation=get_value(nxgroup.attenuation),
            polarization=get_value(nxgroup.polarization),
            inverse_lorentz=get_value(nxgroup.inverse_lorentz),
            d3s=get_value(nxgroup.d3s),
        )

    def __str__(self):
        return str(self.__dict__)


class Symmetry:
    """space group symmetry information"""

    # This is a container class for derived space group info, mainly to handle file format conversion

    def __init__(self, **kwargs):
        # lazy constructor....
        self.__dict__.update(kwargs)

    def is_asu(self, h, k, l):
        return self._test_reflections(h, k, l, self.reciprocal_space_asu)

    def is_reflection(self, h, k, l):
        return self._test_reflections(h, k, l, self.reflection_conditions)

    def _test_reflections(self, h, k, l, nexpr):
        if not isinstance(h, np.ndarray):
            h = np.array(h)
            k = np.array(k)
            l = np.array(l)
        return ne.evaluate(
            nexpr,
            local_dict={"h": h.astype(int), "k": k.astype(int), "l": l.astype(int)},
        )

    def to_asu(self, h, k, l, block=1 << 22):  # noqa: E741
        """Map Miller indices to the asymmetric unit: (h, k, l, op) with op the index of the Laue operator used (the
        lowest index that maps into the asymmetric unit). Rows are processed in blocks (bounded temporaries); the
        operators are applied and the results selected in compiled loops, the asymmetric-unit test is numexpr. Same
        result as to_asu_reference (identical for integer indices); indices that no operator maps into the asymmetric
        unit (an inconsistent symmetry) give 0 here, uninitialized values there."""
        h, k, l = (a if isinstance(a, np.ndarray) else np.array(a) for a in (h, k, l))  # noqa: E741
        shape = h.shape
        out_dtype = np.result_type(h, k, l)
        hs, ks, ls = (np.ascontiguousarray(a).reshape(-1) for a in (h, k, l))
        ops = np.ascontiguousarray(self.laue_group_operators, dtype=np.float64)
        n = hs.size
        ho, ko, lo = (np.zeros(n, dtype=out_dtype) for _ in range(3))
        opindex = np.zeros(n, dtype=int)
        for i0 in range(0, n, block):
            sl = slice(i0, min(i0 + block, n))
            hb, kb, lb = hs[sl], ks[sl], ls[sl]
            for m in reversed(range(ops.shape[0])):
                hn, kn, ln = _apply_laue_op(ops[m], hb, kb, lb)
                _select_asu(self.is_asu(hn, kn, ln), hn, kn, ln, m, ho[sl], ko[sl], lo[sl], opindex[sl])
        return ho.reshape(shape), ko.reshape(shape), lo.reshape(shape), opindex.reshape(shape)

    def to_asu_reference(self, h, k, l):  # noqa: E741
        """mdx2 <= 1.0.5 implementation of to_asu (whole-array temporaries; tensordot and masked assignment per
        operator). Kept for tests."""
        if not isinstance(h, np.ndarray):
            h = np.array(h)
            k = np.array(k)
            l = np.array(l)

        opindex = np.empty_like(h, dtype=int)
        ops = self.laue_group_operators

        hkl_in = np.stack((h, k, l))
        hkl_out = np.empty_like(hkl_in)

        for n in reversed(range(ops.shape[0])):
            op = ops[n, :]
            hkl_n = np.tensordot(op.T, hkl_in, axes=1)
            is_asu = self.is_asu(hkl_n[0, ...], hkl_n[1, ...], hkl_n[2, ...])
            opindex[is_asu] = n
            hkl_out[:, is_asu] = hkl_n[:, is_asu]

        return hkl_out[0, ...], hkl_out[1, ...], hkl_out[2, ...], opindex

    def to_nexus(self):
        return NXgroup(
            name="symmetry",
            space_group_number=self.space_group_number,
            space_group_symbol=self.space_group_symbol,
            crystal_system=self.crystal_system,
            space_group_operators=self.space_group_operators,
            laue_group_number=self.laue_group_number,
            laue_group_symbol=self.laue_group_symbol,
            laue_group_operators=self.laue_group_operators,
            reciprocal_space_asu=self.reciprocal_space_asu,
            reflection_conditions=self.reflection_conditions,  # added
        )

    @staticmethod
    def from_nexus(obj):
        return Symmetry(
            space_group_number=obj.space_group_number.nxvalue,
            space_group_symbol=obj.space_group_symbol.nxvalue,
            crystal_system=obj.crystal_system.nxvalue,
            space_group_operators=obj.space_group_operators.nxvalue,
            laue_group_number=obj.laue_group_number.nxvalue,
            laue_group_symbol=obj.laue_group_symbol.nxvalue,
            laue_group_operators=obj.laue_group_operators.nxvalue,
            reciprocal_space_asu=obj.reciprocal_space_asu.nxvalue,
            reflection_conditions=obj.reflection_conditions.nxvalue,  # added! incompatible with earlier versions
        )

    @staticmethod
    def from_expt(exptfile):
        E = Experiment.from_file(exptfile)
        return Symmetry(
            space_group_number=E.space_group_number,
            space_group_symbol=E.space_group_symbol,
            crystal_system=E.crystal_system,
            space_group_operators=E.space_group_operators,
            laue_group_number=E.laue_group_number,
            laue_group_symbol=E.laue_group_symbol,
            laue_group_operators=E.laue_group_operators,
            reciprocal_space_asu=E.reciprocal_space_asu,
            reflection_conditions=E.reflection_conditions,  # added!
        )

    def __str__(self):
        return str(self.__dict__)


class Crystal:
    """unit cell information"""

    # This is a container class for unit cell info, to handle file format conversion
    #
    # see also:
    #   https://manual.nexusformat.org/classes/base_classes/NXsample.html
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def from_nexus(sample):
        return Crystal(
            space_group=sample.space_group.nxvalue,
            unit_cell=sample.unit_cell.nxvalue,
            orientation_matrix=sample.orientation_matrix.nxvalue,
            ub_matrix=sample.ub_matrix.nxvalue,
        )

    def to_nexus(self):
        return NXsample(
            name="crystal",
            space_group=self.space_group,
            unit_cell=self.unit_cell,
            orientation_matrix=self.orientation_matrix,
            ub_matrix=self.ub_matrix,
        )

    @staticmethod
    def from_expt(exptfile):
        E = Experiment.from_file(exptfile)
        return Crystal(
            space_group=E.space_group,
            unit_cell=E.unit_cell,
            orientation_matrix=E.orientation_matrix,
            ub_matrix=E.ub_matrix,
        )

    def __str__(self):
        return str(self.__dict__)


def peak_expression(peak_model, sigma_cutoff):
    """numexpr string, True where the fractional offsets (h - round(h), ...) lie inside the peak model's contour.

    d = W (dh - r0) with W = inv(sigma) (as GaussianPeak.quadform_at_points); inside if |d|^2 < sigma_cutoff^2. All
    constants are inlined, so the test is one pass over h, k, l without temporaries.
    """
    W = np.linalg.inv(peak_model.sigma)
    r0 = np.asarray(peak_model.r0, dtype=float)
    frac = [f"({v} - floor({v} + 0.5) - ({r!r}))" for v, r in zip("hkl", r0.tolist())]
    d = ["(" + " + ".join(f"({w!r})*{f}" for w, f in zip(row.tolist(), frac)) + ")" for row in W]
    return f"({d[0]}**2 + {d[1]}**2 + {d[2]}**2) < {float(sigma_cutoff) ** 2!r}"


def evaluate_peak_mask(h, k, l, peak_expr, reflection_conditions):  # noqa: E741
    """reflection_conditions(round(h), round(k), round(l)) AND peak_expr(h, k, l), at fractional Miller indices.

    The peak test is one numexpr pass; the reflection conditions are evaluated in integers (as
    Symmetry.is_reflection) only at the few pixels that pass it.
    """
    h, k, l = (np.asarray(a, dtype=np.float64) for a in (h, k, l))  # noqa: E741
    # one numexpr thread: callers run this per chunk in parallel worker processes (no oversubscription)
    nthreads = ne.set_num_threads(1)
    try:
        m = ne.evaluate(peak_expr, local_dict={"h": h, "k": k, "l": l})
        idx = np.nonzero(m)  # index arrays: work for any memory layout (numexpr may return a strided array)
        if idx[0].size:
            H, K, L = (np.round(a[idx]).astype(int) for a in (h, k, l))
            m[idx] = ne.evaluate(reflection_conditions, local_dict={"h": H, "k": K, "l": L})
    finally:
        ne.set_num_threads(nthreads)
    return m


def peak_model_mask(h, k, l, symmetry, peak_model, sigma_cutoff):  # noqa: E741
    """Pixels inside the peak model (sigma_cutoff contour) around allowed reflections: the model part of
    mdx2.mask_peaks, shared by the static (--static) and dynamic masks so that they agree."""
    return evaluate_peak_mask(h, k, l, peak_expression(peak_model, sigma_cutoff), symmetry.reflection_conditions)


class PeakMask:
    """Peak mask stored as two numexpr expressions and a list of pixels, rebuilt chunk by chunk.

    At each pixel, with fractional Miller indices h, k, l:
        excluded = (bragg XOR [reflection_conditions(round(h), round(k), round(l)) AND peak_expression(h, k, l)])
                   OR (frame, iy, ix) in pixels
    peak_expression is built from the peak model (peak_r0, peak_sigma) and sigma_cutoff, which are stored alongside
    for reference; the mask is evaluated from the expressions. pixels: integer (frame, iy, ix) indices on the image
    series (the count-threshold peak pixels the peak model does not cover), sorted. shape: the image-series shape.
    for_chunk(sl) returns a function of the chunk's regridded Miller indices (as computed by ImageSeries.index).
    """

    FORMAT_VERSION = 1

    def __init__(
        self, peak_expression, reflection_conditions, shape, pixels=None, bragg=False,
        peak_r0=None, peak_sigma=None, sigma_cutoff=None,
    ):  # fmt: skip
        self.peak_expression = str(peak_expression)
        self.reflection_conditions = str(reflection_conditions)
        self.shape = tuple(int(n) for n in shape)
        self.bragg = bool(bragg)
        if pixels is None:
            pixels = np.zeros((0, 3), dtype=np.int32)
        pixels = np.asarray(pixels, dtype=np.int32).reshape(-1, 3)
        self.pixels = pixels[np.lexsort((pixels[:, 2], pixels[:, 1], pixels[:, 0]))]
        self.peak_r0 = None if peak_r0 is None else np.asarray(peak_r0, dtype=float)
        self.peak_sigma = None if peak_sigma is None else np.asarray(peak_sigma, dtype=float)
        self.sigma_cutoff = None if sigma_cutoff is None else float(sigma_cutoff)

    @staticmethod
    def from_peak_model(symmetry, peak_model, sigma_cutoff, shape, pixels=None, bragg=False):
        return PeakMask(
            peak_expression(peak_model, sigma_cutoff), symmetry.reflection_conditions, shape, pixels=pixels,
            bragg=bragg, peak_r0=peak_model.r0, peak_sigma=peak_model.sigma, sigma_cutoff=sigma_cutoff,
        )  # fmt: skip

    def model_mask(self, h, k, l):  # noqa: E741
        m = evaluate_peak_mask(h, k, l, self.peak_expression, self.reflection_conditions)
        return ~m if self.bragg else m

    def chunk_pixels(self, sl):
        """Extra pixels inside the chunk sl = (frames, iy, ix) slices, as indices relative to the chunk."""
        a, b = np.searchsorted(self.pixels[:, 0], [sl[0].start, sl[0].stop])
        p = self.pixels[a:b]
        inside = (p[:, 1] >= sl[1].start) & (p[:, 1] < sl[1].stop) & (p[:, 2] >= sl[2].start) & (p[:, 2] < sl[2].stop)
        p = p[inside]
        return p[:, 0] - sl[0].start, p[:, 1] - sl[1].start, p[:, 2] - sl[2].start

    def for_chunk(self, sl):
        """Function mi -> boolean mask (True = excluded) for chunk sl, mi = miller_index.regrid(chunk axes)."""
        sl = tuple(slice(*s.indices(n)) for s, n in zip(sl, self.shape))
        extra = self.chunk_pixels(sl)

        def mask(mi):
            m = self.model_mask(mi.h, mi.k, mi.l)
            m[extra] = True
            return m

        return mask

    def check_axes(self, phi, iy, ix):
        if self.shape != (len(phi), len(iy), len(ix)):
            raise ValueError(f"mask shape {self.shape} does not match the image series {(len(phi), len(iy), len(ix))}")

    def to_nexus(self):
        g = NXgroup(
            name="mask",
            peak_expression=self.peak_expression,
            reflection_conditions=self.reflection_conditions,
            bragg=int(self.bragg),
            pixels=array_field(self.pixels, "pixels"),
            shape=np.array(self.shape, dtype=np.int64),
        )
        for name in ("peak_r0", "peak_sigma", "sigma_cutoff"):
            if getattr(self, name) is not None:
                g[name] = getattr(self, name)
        g.attrs["format_version"] = self.FORMAT_VERSION
        return g

    @staticmethod
    def from_nexus(g):
        version = int(g.attrs.get("format_version", 0))
        if version != PeakMask.FORMAT_VERSION:
            raise ValueError(f"unsupported PeakMask format version {version} (supported: {PeakMask.FORMAT_VERSION})")
        opt = {n: g[n].nxvalue for n in ("peak_r0", "peak_sigma", "sigma_cutoff") if n in g}
        return PeakMask(
            g.peak_expression.nxvalue, g.reflection_conditions.nxvalue, g.shape.nxvalue,
            pixels=g.pixels.nxvalue, bragg=bool(g.bragg.nxvalue), **opt,
        )  # fmt: skip


class StaticMask:
    """Pre-computed boolean mask (mdx2.mask_peaks --static, and all masks from earlier versions), read per chunk."""

    def __init__(self, signal):
        self.signal = signal  # NXfield; pickles as a reference to the file, so workers read only their chunk

    def for_chunk(self, sl):
        sub = self.signal[sl]
        data = sub.nxdata
        sub._value = None  # break the NXfield / AttrDict reference cycle's hold on the array (see ImageSeries)
        return data

    def check_axes(self, phi, iy, ix):
        if tuple(self.signal.shape) != (len(phi), len(iy), len(ix)):
            raise ValueError(f"mask shape {tuple(self.signal.shape)} does not match the image series")


def load_mask(filename, name="mask"):
    """Static (GridData) or dynamic (PeakMask) mask from mdx2.mask_peaks."""
    g = nxload(filename)["entry/" + name]
    if g.attrs.get("mdx2_class") == "PeakMask":
        return loadobj(filename, name)
    return StaticMask(g.signal)  # GridData: read lazily, the array may be very large


@nb.njit(nogil=True, cache=True)
def _apply_laue_op(op, h, k, l):  # noqa: E741
    """Indices transformed by a Laue operator, as np.tensordot(op.T, (h, k, l), axes=1) (float64). Serial and GIL-free
    (callers parallelize over row blocks, e.g. mdx2.correct; no numba threads inside worker processes)."""
    n = h.size
    hn = np.empty(n, dtype=np.float64)
    kn = np.empty(n, dtype=np.float64)
    ln = np.empty(n, dtype=np.float64)
    for i in range(n):
        a, b, c = np.float64(h[i]), np.float64(k[i]), np.float64(l[i])
        hn[i] = op[0, 0] * a + op[1, 0] * b + op[2, 0] * c
        kn[i] = op[0, 1] * a + op[1, 1] * b + op[2, 1] * c
        ln[i] = op[0, 2] * a + op[1, 2] * b + op[2, 2] * c
    return hn, kn, ln


@nb.njit(nogil=True, cache=True)
def _select_asu(mask, hn, kn, ln, m, ho, ko, lo, opindex):
    """Where mask: take the transformed indices (cast to the output dtype, as a numpy assignment) and operator m."""
    for i in range(mask.size):
        if mask[i]:
            ho[i] = hn[i]
            ko[i] = kn[i]
            lo[i] = ln[i]
            opindex[i] = m
