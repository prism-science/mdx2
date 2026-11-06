"""mdx2.correct: the fast engine (row blocks in threads, compiled asymmetric-unit mapping) writes the same table as
the reference engine; Symmetry.to_asu matches to_asu_reference."""

import numpy as np
import pytest

import mdx2.command_line.correct as correct_cmd
from mdx2.command_line.correct import Parameters, run_correct
from mdx2.command_line.integrate import Parameters as IntParams
from mdx2.command_line.integrate import run_integrate
from mdx2.geometry import GridData, Symmetry
from mdx2.io import loadobj, saveobj

from .test_peak_mask import make_masks, write_sweep


@pytest.fixture(scope="module")
def integrated(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("correct")
    tmp_path, geom, dat, pk = write_sweep(tmp_path, chunks=(4, 20, 25))
    _, fd = make_masks(tmp_path, geom, dat, pk)
    hkl = str(tmp_path / "integrated.nxs")
    run_integrate(IntParams(geom=geom, data=dat, mask=fd, subdivide=(2, 2, 2), outfile=hkl))
    ims = loadobj(dat, "image_series")
    rng = np.random.default_rng(3)
    axes = (np.linspace(ims.phi[0], ims.phi[-1], 5), np.linspace(0, 59, 7), np.linspace(0, 69, 8))
    bkg = GridData(axes, rng.random((5, 7, 8)), axes_names=["phi", "iy", "ix"])
    bg = str(tmp_path / "binned.nxs")
    saveobj(bkg, bg, name="binned_image_series")
    return tmp_path, geom, hkl, bg


@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("p1,lorentz", [(False, False), (True, True)])
@pytest.mark.parametrize("nthreads", [1, 3])
def test_engines_identical(integrated, monkeypatch, background, p1, lorentz, nthreads):
    tmp_path, geom, hkl, bg = integrated
    monkeypatch.setattr(correct_cmd, "BLOCK_ROWS", 97)  # many small blocks
    tables = {}
    for engine in ("reference", "fast"):
        out = str(tmp_path / f"corrected_{engine}_{background}_{p1}_{nthreads}.nxs")
        run_correct(
            Parameters(
                geom=geom, hkl=hkl, background=bg if background else None, p1=p1, lorentz=lorentz,
                engine=engine, nthreads=nthreads, outfile=out,
            )
        )  # fmt: skip
        tables[engine] = loadobj(out, "hkl_table")
    a, b = tables["reference"], tables["fast"]
    assert list(a.__dict__) == list(b.__dict__)
    assert len(a) > 200
    for c, v in a.__dict__.items():
        if isinstance(v, np.ndarray):
            w = getattr(b, c)
            assert v.dtype == w.dtype, c
            np.testing.assert_array_equal(v, w, err_msg=c)


def test_to_asu_matches_reference():
    rng = np.random.default_rng(0)
    p1 = "(h > 0) | ((h == 0) & (k > 0)) | ((h == 0) & (k == 0) & (l >= 0))"
    sym = Symmetry(laue_group_operators=np.stack([np.eye(3), -np.eye(3)]), reciprocal_space_asu=p1)
    signs = [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]  # mmm: every index maps into the asu
    mmm = Symmetry(
        laue_group_operators=np.stack([np.diag(d).astype(float) for d in signs]),
        reciprocal_space_asu="(h >= 0) & (k >= 0) & (l >= 0)",
    )
    for s in (sym, mmm):
        for shape in [(5,), (1000,), (30, 40)]:
            H, K, L = (rng.integers(-20, 21, int(np.prod(shape))).reshape(shape) for _ in range(3))
            new, ref = s.to_asu(H, K, L, block=333), s.to_asu_reference(H, K, L)
            for x, y in zip(new, ref):
                assert x.dtype == y.dtype and x.shape == y.shape
                np.testing.assert_array_equal(x, y)
