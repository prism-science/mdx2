"""Dynamic peak masks (mdx2.geometry.PeakMask) must reproduce the static masks of mdx2.mask_peaks --static.

Small synthetic sweep written to NeXus files; mask_peaks both ways (and --bragg), masks compared chunk by chunk,
then mdx2.integrate with each mask must give the same table.
"""

import numpy as np
import pytest
from nexusformat.nexus import NXfield

from mdx2.command_line.integrate import Parameters as IntParams
from mdx2.command_line.integrate import run_integrate
from mdx2.command_line.mask_peaks import Parameters as MaskParams
from mdx2.command_line.mask_peaks import run_mask_peaks
from mdx2.command_line.reintegrate import Parameters as ReintParams
from mdx2.command_line.reintegrate import run_reintegrate
from mdx2.data import ImageSeries, Peaks
from mdx2.geometry import (
    Corrections,
    Crystal,
    GaussianPeak,
    MillerIndex,
    PeakMask,
    StaticMask,
    Symmetry,
    load_mask,
    peak_model_mask,
)
from mdx2.io import loadobj, saveobj

I_CENTRING = "(((h + k + l) % 2) == 0)"
P1_ASU = "(h > 0) | ((h == 0) & (k > 0)) | ((h == 0) & (k == 0) & (l >= 0))"  # with ops [1, -1]


def symmetry():
    return Symmetry(
        space_group_number=197, space_group_symbol="I 2 3", crystal_system="cubic",
        space_group_operators=np.eye(3)[None], laue_group_number=2, laue_group_symbol="-1",
        laue_group_operators=np.stack([np.eye(3), -np.eye(3)]), reciprocal_space_asu=P1_ASU,
        reflection_conditions=I_CENTRING,
    )  # fmt: skip


def write_sweep(tmp_path, chunks=None):
    """Synthetic sweep written to files; chunks: chunk shape of the image data (default: chosen by nexusformat)."""
    rng = np.random.default_rng(1)
    phi, iy, ix = np.arange(0, 8, 0.5), np.arange(60.0), np.arange(70.0)
    # Miller-index lookup grid (coarser than the images, as mdx2.import_geometry), linear in (phi, iy, ix)
    g_phi, g_iy, g_ix = np.linspace(-1, 9, 11), np.linspace(-5, 65, 15), np.linspace(-5, 75, 17)
    P, Y, X = np.meshgrid(g_phi, g_iy, g_ix, indexing="ij")
    h = 0.11 * X + 0.02 * Y + 0.05 * P - 2.0
    k = -0.03 * X + 0.09 * Y + 0.04 * P + 1.0
    l = 0.02 * X + 0.01 * Y - 0.12 * P + 0.5  # noqa: E741
    mi = MillerIndex(g_phi, g_iy, g_ix, h, k, l)
    peak = GaussianPeak(
        np.array([0.01, -0.005, 0.002]), np.array([[0.06, 0.01, 0.0], [0.0, 0.05, 0.01], [0.01, 0.0, 0.07]])
    )
    data = rng.poisson(5.0, (phi.size, iy.size, ix.size)).astype(np.int32)
    data[rng.random(data.shape) < 0.002] = -1  # detector mask value
    hot = rng.random(data.shape) < 0.001
    data[hot] = 500
    ims = ImageSeries(phi, iy, ix, data if chunks is None else NXfield(data, chunks=chunks), np.full(phi.size, 0.1))
    peaks = Peaks.where(hot, phi, iy, ix)
    geom, dat, pk = (str(tmp_path / f) for f in ("geometry.nxs", "data.nxs", "peaks.nxs"))
    saveobj(mi, geom, name="miller_index", append=False)
    saveobj(symmetry(), geom, name="symmetry", append=True)
    # crystal and smooth correction factors (needed by mdx2.reintegrate)
    ub = np.diag([1 / 50.0, 1 / 60.0, 1 / 70.0])
    saveobj(
        Crystal(
            space_group="I 2 3",
            unit_cell=np.array([50.0, 60, 70, 90, 90, 90]),
            orientation_matrix=np.eye(3),
            ub_matrix=ub,
        ),
        geom,
        name="crystal",
        append=True,
    )
    c_iy, c_ix = np.linspace(-5, 65, 8), np.linspace(-5, 75, 9)
    CY, CX = np.meshgrid(c_iy, c_ix, indexing="ij")
    ones = np.ones_like(CY)
    corr = Corrections(
        c_iy,
        c_ix,
        solid_angle=1e-6 * (1 + 0.001 * CX),
        efficiency=0.95 * ones,
        attenuation=0.9 * ones,
        polarization=1 - 0.0005 * CY,
        inverse_lorentz=0.5 * ones,
        d3s=1e-7 * ones,
    )
    saveobj(corr, geom, name="corrections", append=True)
    saveobj(ims, dat, name="image_series", append=False)
    saveobj(peak, pk, name="peak_model", append=False)
    saveobj(peaks, pk, name="peaks", append=True)
    return tmp_path, geom, dat, pk


@pytest.fixture
def sweep(tmp_path):
    return write_sweep(tmp_path)


def make_masks(tmp_path, geom, dat, pk, bragg=False):
    out = {}
    for static in (True, False):
        f = str(tmp_path / f"mask_{'static' if static else 'dynamic'}_{int(bragg)}.nxs")
        run_mask_peaks(
            MaskParams(geom=geom, data=dat, peaks=pk, outfile=f, static=static, bragg=bragg, sigma_cutoff=2.0)
        )
        out[static] = f
    return out[True], out[False]


@pytest.mark.parametrize("bragg", [False, True])
def test_dynamic_equals_static(sweep, bragg):
    tmp_path, geom, dat, pk = sweep
    fs, fd = make_masks(tmp_path, geom, dat, pk, bragg=bragg)
    S, D = load_mask(fs), load_mask(fd)
    assert isinstance(S, StaticMask) and isinstance(D, PeakMask)
    mi, ims = loadobj(geom, "miller_index"), loadobj(dat, "image_series")
    n_masked = 0
    for sl in ims.chunk_slice_iterator():
        a = S.for_chunk(sl)
        b = D.for_chunk(sl)(mi.regrid(ims.phi[sl[0]], ims.iy[sl[1]], ims.ix[sl[2]]))
        np.testing.assert_array_equal(a, b)
        n_masked += a.sum()
    assert 0 < n_masked < (ims.data.size if not bragg else ims.data.size + 1)
    if not bragg:
        assert len(D.pixels) > 0  # some hot pixels lie outside the peak model and are stored


def test_integrate_same_table(sweep):
    tmp_path, geom, dat, pk = sweep
    fs, fd = make_masks(tmp_path, geom, dat, pk)
    tabs = []
    for f in (fs, fd):
        out = str(tmp_path / f"int_{len(tabs)}.nxs")
        run_integrate(IntParams(geom=geom, data=dat, mask=f, subdivide=(2, 2, 2), outfile=out))
        tabs.append(loadobj(out, "hkl_table"))
    a, b = tabs
    assert len(a) == len(b) > 0
    for c in ("h", "k", "l", "counts", "pixels", "seconds", "phi", "ix", "iy"):
        np.testing.assert_array_equal(getattr(a, c), getattr(b, c), err_msg=c)


@pytest.mark.parametrize("output", ["counts", "intensity"])
def test_reintegrate_same_table(sweep, output):
    tmp_path, geom, dat, pk = sweep
    fs, fd = make_masks(tmp_path, geom, dat, pk)
    tabs = []
    for f in (fs, fd):
        out = str(tmp_path / f"reint_{len(tabs)}.nxs")
        run_reintegrate(ReintParams(geom=geom, data=dat, mask=f, subdivide=(2, 2, 2), output=output, outfile=out))
        tabs.append(loadobj(out, "hkl_table"))
    a, b = tabs
    assert len(a) == len(b) > 0
    for c in ["h", "k", "l"] + sorted(a._data_keys):
        np.testing.assert_array_equal(getattr(a, c), getattr(b, c), err_msg=c)


def test_mask_shape_mismatch(sweep):
    tmp_path, geom, dat, pk = sweep
    _, fd = make_masks(tmp_path, geom, dat, pk)
    D = load_mask(fd)
    phi, iy, ix = (np.arange(n) for n in D.shape)
    D.check_axes(phi, iy, ix)
    with pytest.raises(ValueError, match="does not match"):
        D.check_axes(phi[:-1], iy, ix)


def test_file_is_self_describing(sweep):
    """The mask file holds the numexpr expressions; evaluating them by hand (rule in the PeakMask docstring)
    reproduces the mask."""
    import numexpr as ne  # noqa: PLC0415

    from mdx2.io import nxload  # noqa: PLC0415

    tmp_path, geom, dat, pk = sweep
    _, fd = make_masks(tmp_path, geom, dat, pk)
    g = nxload(fd)["entry/mask"]
    assert g.sigma_cutoff.nxvalue == 2.0
    D = load_mask(fd)
    mi, ims = loadobj(geom, "miller_index"), loadobj(dat, "image_series")
    sl = next(iter(ims.chunk_slice_iterator()))
    m = mi.regrid(ims.phi[sl[0]], ims.iy[sl[1]], ims.ix[sl[2]])
    h, k, l = m.h, m.k, m.l  # noqa: E741
    inside = ne.evaluate(str(g.peak_expression.nxvalue))
    H, K, L = (np.round(a).astype(int) for a in (h, k, l))
    allowed = ne.evaluate(str(g.reflection_conditions.nxvalue), local_dict={"h": H, "k": K, "l": L})
    by_hand = allowed & inside
    for f, y, x in g.pixels.nxvalue:
        if all(s.start <= v < s.stop for s, v in zip(sl, (f, y, x))):
            by_hand[f - sl[0].start, y - sl[1].start, x - sl[2].start] = True
    np.testing.assert_array_equal(by_hand, D.for_chunk(sl)(m))


def test_ellipsoid_expression_matches_quadform():
    rng = np.random.default_rng(3)
    peak = GaussianPeak(rng.normal(size=3) * 0.01, rng.normal(size=(3, 3)) * 0.05)
    h, k, l = (rng.integers(-20, 20, 20000) + rng.normal(0, 0.08, 20000) for _ in range(3))  # noqa: E741
    H, K, L = np.round(h), np.round(k), np.round(l)
    ref = symmetry().is_reflection(H, K, L) & peak.mask(h - H, k - K, l - L, sigma_cutoff=2.5)
    got = peak_model_mask(h.reshape(100, 200), k.reshape(100, 200), l.reshape(100, 200), symmetry(), peak, 2.5)
    assert ref.sum() > 100
    np.testing.assert_array_equal(got.ravel(), ref)
