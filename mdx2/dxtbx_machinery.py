import math
import re

import h5py
import hdf5plugin  # noqa: F401  (HDF5 filters for detector data, e.g. bitshuffle, in NXmxFrameReader)
import numpy as np
from cctbx.eltbx import attenuation_coefficient
from dxtbx import flumpy
from dxtbx.format.FormatNXmx import FormatNXmx
from dxtbx.format.FormatPilatusHelpers import sensor_active_areas
from dxtbx.model.experiment_list import ExperimentList
from scitbx import matrix


class Experiment:
    """Wrapper for dxtbx Experiment object"""

    def __init__(self, expt):
        self._crystal = expt.crystal
        self._goniometer = expt.goniometer
        self._panel = expt.detector[0]  # single-panel detectors assumed
        self._scan = expt.scan
        self._beam = expt.beam

        # check compatibility
        try:
            # Check that we have exactly one panel
            if len(expt.detector) != 1:
                raise RuntimeError(
                    f"Experiment detector has {len(expt.detector)} panels; mdx2 requires single-panel detectors"
                )

            # Check that the panel type is SENSOR_PAD
            if self._panel.get_type() != "SENSOR_PAD":
                raise RuntimeError(
                    f"Experiment detector type is '{self._panel.get_type()}'; mdx2 requires SENSOR_PAD detectors"
                )

            # Check that the detector is in the known database using public API
            active_areas = sensor_active_areas((self._panel,))
            if active_areas is None or len(active_areas) == 0:
                raise RuntimeError(
                    "Experiment detector geometry not found in dxtbx detector database; "
                    "mdx2 may not be compatible with this detector type"
                )

            # Extract the first (and only) active area tuple
            active_area = active_areas[0]
            # Note: active_area is (fast_start, slow_start, fast_end, slow_end)
            fast_start, slow_start, fast_end, slow_end = active_area
            image_size = self._panel.get_image_size()  # (fast, slow)

            # Module size is the active area dimension
            self._module_size_slow = slow_end - slow_start
            self._module_size_fast = fast_end - fast_start

            # Calculate number of modules in each direction
            # image_size is (fast, slow), so image_size[1] is slow, image_size[0] is fast
            num_modules_slow = image_size[1] // self._module_size_slow
            num_modules_fast = image_size[0] // self._module_size_fast

            # Calculate gap sizes
            if num_modules_slow > 1:
                total_gap_slow = image_size[1] - (num_modules_slow * self._module_size_slow)
                self._gap_slow = total_gap_slow // (num_modules_slow - 1)
            else:
                self._gap_slow = 0

            if num_modules_fast > 1:
                total_gap_fast = image_size[0] - (num_modules_fast * self._module_size_fast)
                self._gap_fast = total_gap_fast // (num_modules_fast - 1)
            else:
                self._gap_fast = 0

        except (AssertionError, RuntimeError) as e:
            raise RuntimeError(f"Experiment detector not compatible with mdx2: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error checking detector compatibility: {e}") from e

    @staticmethod
    def from_file(exptfile):
        elist = ExperimentList.from_file(exptfile)
        expt = elist[0]  # single experiment in file assumed
        return Experiment(expt)

    @property
    def image_size(self):
        nx, ny = self._panel.get_image_size()
        return nx, ny

    @property
    def oscillation_range(self):
        phimin, phimax = self._scan.get_oscillation_range()
        return phimin, phimax

    @property
    def oscillation(self):
        phi0, dphi = self._scan.get_oscillation()
        return phi0, dphi

    @property
    def num_images(self):
        nimgs = self._scan.get_num_images()
        return nimgs

    @property
    def panel_offset(self):
        """offset in pixels (stride) of successive panels in the image file"""
        row_size = self._module_size_slow + self._gap_slow
        col_size = self._module_size_fast + self._gap_fast
        return row_size, col_size

    @property
    def exposure_times(self):
        dt = np.array(tuple(self._scan.get_exposure_times()))
        # NOTE: different versions of dxtbx encode exposure times differently.
        # TODO: implement checks for dxtbx version mismatchs and resolve before errors occur.

        # Fix a bug in dxtbx. When cbf files are imported without reading all the headers,
        # only the first element in the list of exposure times is set, and the rest are zero.
        if np.count_nonzero(dt) == 1 and dt[0] > 0:
            dt[:] = dt[0]
        return dt

    @property
    def space_group_number(self):
        symm = self._crystal.get_crystal_symmetry()
        sg = symm.space_group()
        sgi = sg.info()
        sgt = sgi.type()
        return sgt.number()

    @property
    def space_group_symbol(self):
        symm = self._crystal.get_crystal_symmetry()
        sg = symm.space_group()
        sgi = sg.info()
        sgt = sgi.type()
        return sgt.lookup_symbol()

    @property
    def crystal_system(self):
        symm = self._crystal.get_crystal_symmetry()
        sg = symm.space_group()
        return sg.crystal_system()

    @property
    def reciprocal_space_asu(self):
        symm = self._crystal.get_crystal_symmetry()
        sg = symm.space_group()
        sgi = sg.info()
        asu = sgi.reciprocal_space_asu()
        asu_str = asu.reference_as_string()
        # make substitutions for numpy compatibility
        asu_str = re.sub(r"([hkl0\<\>\=]+)", r"(\1)", asu_str)
        asu_str = re.sub("and", r"&", asu_str)
        asu_str = re.sub("or", r"|", asu_str)
        # apply a change-of-basis operation if needed
        # see: https://github.com/cctbx/cctbx_project/blob/master/cctbx/sgtbx/reciprocal_space_asu.h
        if not asu.is_reference():
            op = asu.cb_op().as_hkl()
            hklmap = {k: f"({v})" for k, v in zip(["h", "k", "l"], op.split(","))}
            asu_str = re.sub("(h|k|l)", lambda x: hklmap[x.group()], asu_str)
        return asu_str

    @property
    def space_group_operators(self):
        symm = self._crystal.get_crystal_symmetry()
        sg = symm.space_group()
        ops = sg.all_ops()

        def op2np(op):
            r = np.array(op.r().as_double()).reshape(3, 3)
            t = np.array(op.t().as_double()).reshape(3, 1)
            return np.hstack((r, t))

        sg_ops = np.stack([op2np(op) for op in ops], axis=0)
        return sg_ops

    @property
    def laue_group_number(self):
        symm = self._crystal.get_crystal_symmetry()
        lg = symm.space_group().build_derived_laue_group()
        return lg.info().type().number()

    @property
    def laue_group_symbol(self):
        symm = self._crystal.get_crystal_symmetry()
        return symm.space_group().laue_group_type()

    @property
    def laue_group_operators(self):
        symm = self._crystal.get_crystal_symmetry()
        lg = symm.space_group().build_derived_laue_group()
        ops = lg.all_ops()
        return np.stack([np.array(op.r().as_double()).reshape(3, 3) for op in ops], axis=0)

    @property
    def ub_matrix(self):
        return np.array(self._crystal.get_A()).reshape(3, 3)

    @property
    def space_group(self):
        sg = self._crystal.get_space_group()
        return sg.info().type().lookup_symbol()

    @property
    def orientation_matrix(self):
        return np.array(self._crystal.get_U()).reshape(3, 3)

    @property
    def unit_cell(self):
        uc = self._crystal.get_unit_cell()
        return uc.parameters()

    @property
    def scan_axes(self):
        return self.calc_scan_axes()

    @property
    def reflection_conditions(self):
        sg = self._crystal.get_space_group()
        centering_type = sg.conventional_centring_type_symbol()
        return lookup_reflection_conditions(centering_type)

    def calc_scan_axes(self, centered=True, samples=None, spacing=None):
        nx, ny = self.image_size
        nimgs = self.num_images
        phi0, dphi = self.oscillation

        if samples is not None:
            # use samples as provided
            pass
        elif spacing is not None:
            # compute samples from spacing
            def ceil(n, d):
                return -1 * int(-n // d)

            samp_x = ceil(nx, spacing[2])
            samp_y = ceil(ny, spacing[1])
            samp_phi = ceil(dphi * nimgs, spacing[0])
            if centered:
                samples = (samp_phi, samp_y, samp_x)
            else:
                samples = (1 + samp_phi, 1 + samp_y, 1 + samp_x)
        else:
            if centered:
                samples = (nimgs, ny, nx)
            else:
                samples = (nimgs + 1, ny + 1, nx + 1)

        if centered:
            phi = phi0 + dphi * np.linspace(0.5, nimgs - 0.5, samples[0])
            iy = np.linspace(0, ny - 1, samples[1])
            ix = np.linspace(0, nx - 1, samples[2])
        else:  # edges
            phi = phi0 + dphi * np.linspace(0, nimgs, samples[0])
            iy = np.linspace(-0.5, ny - 0.5, samples[1])
            ix = np.linspace(-0.5, nx - 0.5, samples[2])

        return phi, iy, ix

    def calc_xyz_on_grid(self, iy, ix):
        x, y, z = index_grid_to_lab(self._panel, ix, iy)
        return x, y, z  # shape? -- TODO: split into x, y, z

    def calc_s_on_grid(self, *args):
        x, y, z = self.calc_xyz_on_grid(*args)
        xyz = np.stack((x, y, z))
        d = np.sqrt(np.sum(xyz * xyz, axis=0))
        wlen = self._beam.get_wavelength()
        s1 = xyz / d / wlen
        s0 = np.array(self._beam.get_s0())
        s = s1 - s0[:, np.newaxis, np.newaxis]
        sx, sy, sz = s[0, :, :], s[1, :, :], s[2, :, :]
        return sx, sy, sz  # shape? -- TODO: split into sx, sy, sz

    def calc_hkl_on_grid(self, phi, *args):
        sx, sy, sz = self.calc_s_on_grid(*args)
        s = np.stack((sx, sy, sz))

        R = calc_rotation_matrix_at_phi(self._goniometer, phi)
        B = calc_B_matrix_at_phi(self._crystal, self._scan, phi)
        U = calc_U_matrix_at_phi(self._crystal, self._scan, phi)

        # array of orthogonalization matrices
        UB = R @ U @ B

        # array of fractionalization matrix
        invUB = np.linalg.inv(UB)

        hkl = np.tensordot(invUB, s, axes=1)
        h, k, l = hkl[:, 0, :, :], hkl[:, 1, :, :], hkl[:, 2, :, :]

        return h, k, l

    def calc_corrections_on_grid(self, *args):
        x, y, z = self.calc_xyz_on_grid(*args)
        xyz = np.dstack((x, y, z))

        # some helpful geometry that I'll reuse
        d = np.sqrt(np.sum(xyz * xyz, axis=2))
        dnorm = self._panel.get_normal()
        cosw = np.dot(xyz, dnorm) / d

        solid_angle = calc_solid_angle(self._panel, cosw, d)
        efficiency = calc_efficiency(self._panel, cosw)
        attenuation = calc_attenuation(self._beam, d)
        polarization = calc_polarization(self._beam, xyz, d)
        inverse_lorentz = calc_inverse_lorentz(self._beam, self._goniometer, xyz, d)
        d3s = calc_d3s(self._beam, self._scan, solid_angle, inverse_lorentz)

        return {
            "solid_angle": solid_angle,
            "efficiency": efficiency,
            "attenuation": attenuation,
            "polarization": polarization,
            "inverse_lorentz": inverse_lorentz,
            "d3s": d3s,
        }


class ImageSet:
    """Wrapper for dxtbx imageset object"""

    def __init__(self, iset):
        self._iset = iset

    @staticmethod
    def from_file(exptfile):
        elist = ExperimentList.from_file(exptfile)
        iset = elist.imagesets()[0]  # read only the first image set if multiple
        return ImageSet(iset)

    @property
    def num_frames(self):
        scan = self._iset.get_scan()
        return scan.get_num_images()

    @property
    def frame_shape(self):
        det = self._iset.get_detector()[0]  # <-- single panel assumed
        nx, ny = det.get_image_size()
        return (ny, nx)

    def read_frame(self, ind, maskval=-1):
        # for now, apply mask by default and return just the image as an ndarray
        im = self._iset.get_raw_data(ind)[0]
        msk = self._iset.get_mask(ind)[0]
        msk = ~msk
        if maskval is not None:
            im.set_selected(msk, maskval)  # flex array magic
            image = im.as_numpy_array()
        else:
            image = np.ma.masked_array(im.as_numpy_array(), mask=msk.as_numpy_array())
        return image

    def read_stack(self, start, stop):
        ny, nx = self.frame_shape
        stack = np.empty((stop - start, ny, nx), dtype=np.int32)  # filled in place (no list + np.stack copy)
        for i, ind in enumerate(range(start, stop)):
            stack[i] = self.read_frame(ind)
        return stack

    def direct_reader(self):
        """A NXmxFrameReader for this image set, or (None, reason) if it does not apply."""
        return NXmxFrameReader.from_imageset(self._iset)


def _open_hdf5(path, work, keep_open=False):
    """Open an HDF5 file read-only and return (file, work(file)), with the SWMR flag of any handle already open on the
    same file in this process: HDF5 refuses a second open with a different flag, which shows up when following links
    to the data files. dxtbx opens NXmx files with swmr=True (FormatNXmx); we don't assume it."""
    error = None
    for swmr in (True, False):
        f = None
        try:
            f = h5py.File(path, "r", swmr=swmr)
            result = work(f)
            if not keep_open:
                f.close()
            return f, result
        except (OSError, KeyError) as e:
            if f is not None:
                f.close()
            if "SWMR" not in str(e):
                raise
            error = e
    raise error


class NXmxFrameReader:
    """Reads frames of an NXmx (e.g. Eiger) image set with h5py and applies the dxtbx mask in numpy.

    Same result as ImageSet.read_frame (int32, masked pixels -1) for formats where dxtbx's mask is the static mask
    (from the format and the experiment's external lookup) and the panel's trusted range, inclusive at both ends, as
    dxtbx.model.Panel.get_trusted_range_mask. dxtbx's per-frame overhead (flex arrays, mask objects) is ~2.5x the
    decompression time on Eiger 16M. Picklable (file handles are opened in each process on first use), and needs only
    h5py in the workers: import_data checks it against dxtbx once, in the main process (check_against).
    """

    def __init__(self, master, links, counts, indices, frame_shape, trusted_range, static_mask):
        self.master = master
        self.links = links  # entry/data/data_NNNNNN paths in the master file, in frame order
        self.starts = np.concatenate([[0], np.cumsum(counts)])
        self.indices = np.asarray(indices)  # image-set index -> frame index in the data files
        self.frame_shape = frame_shape
        self.trusted_range = trusted_range
        self.static_mask = static_mask  # bool (ny, nx), True = valid; or None
        self._datasets = None
        self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_datasets"] = None
        state["_file"] = None
        return state

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file = self._datasets = None

    @staticmethod
    def from_imageset(iset):
        """(reader, None) if the direct reader applies to this dxtbx image set, else (None, reason)."""
        fmt = iset.get_format_class()
        if fmt is None or not issubclass(fmt, FormatNXmx):
            return None, f"format {getattr(fmt, '__name__', fmt)} is not NXmx"
        paths = list(iset.paths())
        if len(set(paths)) != 1:
            return None, "more than one file"
        if iset.has_dynamic_mask():
            return None, "dynamic mask"
        detector = iset.get_detector()
        if len(detector) != 1:
            return None, "multi-panel detector"
        panel = detector[0]
        nx, ny = panel.get_image_size()

        def layout(f):
            if "entry/data" not in f:
                return "no entry/data group"
            links = sorted(k for k in f["entry/data"] if re.fullmatch(r"data_\d{6}", k))
            if not links:
                return "no data_NNNNNN datasets"
            shapes = [(k, f["entry/data"][k].shape, f["entry/data"][k].dtype) for k in links]
            return links, shapes

        _, found = _open_hdf5(paths[0], layout)
        if isinstance(found, str):
            return None, found
        links, shapes = found
        counts, dtypes = [], set()
        for k, shape, dt in shapes:
            if len(shape) != 3 or shape[1:] != (ny, nx):
                return None, f"{k} has shape {shape}, panel is {(ny, nx)}"
            counts.append(shape[0])
            dtypes.add(dt)
        if len(dtypes) != 1 or not np.issubdtype(next(iter(dtypes)), np.integer):
            return None, f"data types {dtypes}"
        dtype = next(iter(dtypes))
        lo, hi = panel.get_trusted_range()
        # dxtbx maps the top two values of the readout to -1 / -2 and reads 32-bit data as signed: the same pixels are
        # masked here only if those values are outside the trusted range
        top = 2**31 if dtype.itemsize >= 4 else 2 ** (8 * dtype.itemsize)
        if lo < 0 or hi >= top - 2:
            return None, f"trusted range {(lo, hi)} overlaps dxtbx's special values"
        indices = list(iset.indices())
        if max(indices) >= sum(counts):
            return None, "image indices beyond the data files"
        static = None
        masks = []
        instance = fmt.get_instance(paths[0])
        fmask = instance.get_static_mask()
        if fmask is not None:
            masks.append(fmask[0].as_numpy_array())
        ext = iset.external_lookup.mask
        if ext.data is not None and not ext.data.empty():
            masks.append(ext.data.tile(0).data().as_numpy_array())
        if masks:
            static = np.logical_and.reduce(masks)
        return NXmxFrameReader(paths[0], links, counts, indices, (ny, nx), (lo, hi), static), None

    def _open(self):
        self._file, self._datasets = _open_hdf5(
            self.master, lambda f: [f["entry/data"][k] for k in self.links], keep_open=True
        )
        dtype = self._datasets[0].dtype
        info = np.iinfo(dtype)
        lo, hi = self.trusted_range
        # integer bounds (dxtbx compares in double precision; the same for integer values)
        self.lo = dtype.type(max(math.ceil(lo), info.min))
        self.hi = dtype.type(min(math.floor(hi), info.max))
        self._buf = np.empty(self.frame_shape, dtype=dtype)

    def read_raw(self, ind):
        """Frame ind of the image set as stored (no mask applied; the buffer is reused by the next call).
        Valid pixels: lo <= value <= hi and static_mask (if not None)."""
        if self._datasets is None:
            self._open()
        g = int(self.indices[ind])
        k = int(np.searchsorted(self.starts, g, side="right")) - 1
        self._datasets[k].read_direct(self._buf, np.s_[g - self.starts[k]])
        return self._buf

    def read_frame(self, ind, out=None):
        """Frame ind of the image set as int32 with masked pixels -1 (as ImageSet.read_frame)."""
        v = self.read_raw(ind)
        invalid = (v < self.lo) | (v > self.hi)
        if self.static_mask is not None:
            invalid |= ~self.static_mask
        if out is None:
            out = np.empty(self.frame_shape, dtype=np.int32)
        np.copyto(out, v, casting="unsafe")  # valid values are within the trusted range, so they fit int32
        np.putmask(out, invalid, -1)
        return out

    def check_against(self, imageset):
        """Compare the first frame with ImageSet.read_frame (dxtbx). Returns None if identical, else a description."""
        ref = imageset.read_frame(0)
        mine = self.read_frame(0)
        if not np.array_equal(ref, mine):
            return f"frame 0: {int(np.sum(ref != mine))} pixels differ"
        return None


def calc_rotation_matrix_at_phi(goniometer, phi_vals_deg):
    if goniometer.num_scan_points != 0:
        raise NotImplementedError("scan-varying goniometer not supported here")
    # get the S matrix
    S = goniometer.get_setting_rotation()
    S = np.array(S).reshape([3, 3])
    # get the F matrix
    F = goniometer.get_fixed_rotation()
    F = np.array(F).reshape([3, 3])
    # get the R matrix at each angle
    axis = matrix.col(goniometer.get_rotation_axis_datum())
    R = np.empty([len(phi_vals_deg), 3, 3])
    for ind, phi in enumerate(phi_vals_deg):
        Rvals = tuple(axis.axis_and_angle_as_r3_rotation_matrix(phi, deg=True))
        R[ind, :, :] = np.array(Rvals).reshape(3, 3)
    # multiply
    return S @ R @ F


def interp_rotation_matrices(U1, U2, rotation_fraction):
    # U1, U2 represented as a (flat) tuple, returns a (flat) tuple
    # compute the operator that rotates from from U1 to U2
    M = matrix.sqr(U2) * matrix.sqr(U1).transpose()
    Mq = M.r3_rotation_matrix_as_unit_quaternion()
    angle, axis = Mq.unit_quaternion_as_axis_and_angle(deg=False)
    # interpolate the rotation
    M_frac = axis.axis_and_angle_as_r3_rotation_matrix(angle * rotation_fraction, deg=False)
    return tuple(M_frac * matrix.sqr(U1))


def calc_U_matrix_at_phi(crystal, scan, phi_vals_deg):
    if crystal.num_scan_points == 0:
        U = np.array(crystal.get_U()).reshape(3, 3)
        U = np.tile(U, (np.size(phi_vals_deg), 1, 1))
    else:
        base, frac = phi_to_base_fraction_index(scan, phi_vals_deg)
        imax = crystal.num_scan_points
        U = np.empty([np.size(base), 3, 3])
        for ind, (b, f) in enumerate(zip(base, frac)):
            U1 = crystal.get_U_at_scan_point(b)
            # Handle boundary: clamp second index to valid range
            b2 = min(b + 1, imax - 1)
            if b2 == b:
                # At the boundary, use U1 directly without interpolation
                Uvals = U1
            else:
                U2 = crystal.get_U_at_scan_point(b2)
                Uvals = interp_rotation_matrices(U1, U2, f)
            U[ind, :, :] = np.array(Uvals).reshape(3, 3)
    return U


def phi_to_base_fraction_index(scan, phi_vals_deg):
    # get integer scan index and fractional part for each angle
    # the fractional part is between zero and one for phi angles within the
    # scan range. the integer part can range from 0,(n-1) where n is the
    # maximum array index

    # imin, imax = scan.get_array_range()
    imin = 0
    imax = scan.get_num_images()
    phi0, dphi = scan.get_oscillation()
    b = []
    f = []
    for ind, phi in enumerate(phi_vals_deg):
        # scan_index = scan.get_array_index_from_angle(phi)
        scan_index = (phi - phi0) / dphi
        scan_index_base = int(scan_index)
        if scan_index_base >= imax:
            scan_index_base = imax - 1
        elif scan_index_base < imin:
            scan_index_base = imin
        b.append(scan_index_base)
        f.append(scan_index - scan_index_base)
    return b, f


def calc_B_matrix_at_phi(crystal, scan, phi_vals_deg):
    if crystal.num_scan_points == 0:
        B = np.array(crystal.get_B()).reshape(3, 3)
        B = np.tile(B, (np.size(phi_vals_deg), 1, 1))
    else:
        base, frac = phi_to_base_fraction_index(scan, phi_vals_deg)
        imax = crystal.num_scan_points
        B = np.empty([np.size(base), 3, 3])
        for ind, (b, f) in enumerate(zip(base, frac)):
            B1 = np.array(crystal.get_B_at_scan_point(b)).reshape([3, 3])
            # Handle boundary: clamp second index to valid range
            b2 = min(b + 1, imax - 1)
            if b2 == b:
                # At the boundary, use B1 directly without interpolation
                B[ind, :, :] = B1
            else:
                B2 = np.array(crystal.get_B_at_scan_point(b2)).reshape([3, 3])
                B[ind, :, :] = B1 * (1 - f) + B2 * f  # element-wise linear interpolation
    return B


def index_grid_to_lab(panel, ix, iy):
    xy = np.dstack(np.meshgrid(ix, iy))  # ny-by-nx-by-2
    sz = xy.shape[:-1]
    xy = flumpy.vec_from_numpy(np.double(xy))
    xymm = panel.pixel_to_millimeter(xy)
    xyz = panel.get_lab_coord(xymm)
    xyz = xyz.as_numpy_array()
    xyz = xyz.reshape(sz + (3,))
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    return x, y, z


# corrections
def calc_mu_air(wavelength):
    """approximate attenuation coefficient of air"""
    # (mdx-lib) geom.MaterialProperties.calcMu('Air',1.0) --> 3.0012e-04
    # (this function) mu_air(1.0) --> 3.328e-04
    att_N = attenuation_coefficient.get_table("N")
    att_O = attenuation_coefficient.get_table("O")
    att_Ar = attenuation_coefficient.get_table("Ar")
    mu_N = att_N.mu_at_angstrom(wavelength) / 10.0
    mu_O = att_O.mu_at_angstrom(wavelength) / 10.0
    mu_Ar = att_Ar.mu_at_angstrom(wavelength) / 10.0
    molfrac_O2 = 0.210
    molfrac_N2 = 0.781
    molfrac_Ar = 0.009
    return molfrac_O2 * mu_O + molfrac_N2 * mu_N + molfrac_Ar * mu_Ar  # approximately right


def calc_solid_angle(panel, cosw, d):
    # The solid angle in units of rad^2
    qx, qy = panel.get_pixel_size()
    return qx * qy * cosw / (d * d)


def calc_efficiency(panel, cosw):
    # efficiency factor
    mu = panel.get_mu()
    t = panel.get_thickness()
    return 1 - np.exp(-mu * t / cosw)


def calc_attenuation(beam, d):
    # attenuation due to air
    wlen = beam.get_wavelength()
    mu_air = calc_mu_air(wlen)
    return np.exp(-mu_air * d)


def calc_polarization(beam, xyz, d):
    # polarization factor
    e0 = beam.get_unit_s0()
    pn = beam.get_polarization_normal()
    p = beam.get_polarization_fraction()
    pv1 = np.cross(e0, pn)
    pv2 = np.cross(pv1, e0)
    cos_phi1 = np.dot(xyz, pv1) / d
    cos_phi2 = np.dot(xyz, pv2) / d
    return p * (1 - cos_phi1 * cos_phi1) + (1 - p) * (1 - cos_phi2 * cos_phi2)


def calc_inverse_lorentz(beam, goniometer, xyz, d):
    m2 = goniometer.get_rotation_axis()
    e0 = beam.get_unit_s0()
    return np.abs(np.dot(np.cross(e0, xyz), m2) / d)


def calc_d3s(beam, scan, solidAngle, Linv):
    # volume of reciprocal space swept by a pixel during a rotation frame
    phi0, dphi = scan.get_oscillation()
    wlen = beam.get_wavelength()
    dphi_radians = dphi * np.pi / 180
    return Linv * solidAngle * dphi_radians / wlen**3


def lookup_reflection_conditions(centering_type):
    # reflection conditions due to centering
    # see: https://dictionary.iucr.org/Reflection_conditions

    if centering_type == "P":
        return "True"
    elif centering_type == "F":
        # h + k, h + l and k + l = 2n or: h, k, l all odd or all even
        return "(((h + k) % 2 == 0) & ((h + l) % 2 == 0) & ((k + l) % 2 == 0)) | ((h % 2 == k % 2) & (h % 2 == l % 2))"
    elif centering_type == "I":
        # h + k + l = 2n
        return "(h + k + l) % 2 == 0"
    elif centering_type == "A":
        # k + l = 2n
        return "(k + l) % 2 == 0"
    elif centering_type == "B":
        # l + h = 2n
        return "(l + h) % 2 == 0"
    elif centering_type == "C":
        # h + k = 2n
        return "(h + k) % 2 == 0"
    elif centering_type == "D":
        # h + k + l = 3n
        return "(h + k + l) % 3 == 0"
    elif centering_type == "H":
        # h − k = 3n
        return "(h - k) % 3 == 0"
    elif centering_type == "R":
        # − h + k + l = 3n (standard obverse setting)
        return "(- h + k + l) % 3 == 0"
    else:
        # centering type not recognized!
        return None
