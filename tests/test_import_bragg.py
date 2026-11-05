"""mdx2.import_bragg: argument parsing; on real data when MDX2_TEST_REFL and MDX2_TEST_EXPT are set."""

import os

import numpy as np
import pytest

from mdx2.command_line.import_bragg import Parameters, parse_arguments, run_import_bragg
from mdx2.dials_refl import read_refl
from mdx2.io import loadobj


def test_parse_arguments():
    p = parse_arguments(["integrated.refl", "integrated.expt", "--experiment", "1", "--background", "binned.nxs"])
    assert (p.refl, p.expt, p.experiment, p.background, p.outfile) == (
        "integrated.refl", "integrated.expt", 1, "binned.nxs", "bragg.nxs",
    )  # fmt: skip
    p = parse_arguments(["a.refl", "a.expt"])
    assert p.experiment is None and p.background is None


REFL, EXPT = os.environ.get("MDX2_TEST_REFL"), os.environ.get("MDX2_TEST_EXPT")


@pytest.mark.skipif(not (REFL and EXPT), reason="set MDX2_TEST_REFL and MDX2_TEST_EXPT (integrated.refl / .expt)")
def test_real_data(tmp_path):
    out = tmp_path / "bragg.nxs"
    run_import_bragg(Parameters(refl=REFL, expt=EXPT, outfile=str(out)))
    T = loadobj(str(out), "hkl_table")
    R = read_refl(REFL)
    assert len(T) == int(np.count_nonzero(R.has_flag("integrated_sum")))
    # mdx2's correction factors at the predicted positions reproduce DIALS lp and qe
    np.testing.assert_allclose(T.inverse_lorentz / T.polarization, T.lp, rtol=1e-4)
    np.testing.assert_allclose(T.efficiency, T.qe, rtol=1e-4)
    # intensity = prefactor x I_sum x lp / (qe x attenuation), the prefactor positive and nearly constant
    pref = T.intensity / (T.intensity_sum * T.lp / (T.qe * T.attenuation))
    ok = np.isfinite(pref) & (np.abs(T.intensity_sum) > 1)
    assert np.all(pref[ok] > 0) and np.ptp(pref[ok]) / np.median(pref[ok]) < 0.01
    np.testing.assert_allclose(
        T.intensity_error / T.intensity, np.sqrt(T.intensity_sum_variance) / T.intensity_sum, rtol=1e-4
    )
    for c in (
        "intensity",
        "intensity_error",
        "background_intensity",
        "phi",
        "ix",
        "iy",
        "s",
        "attenuation",
        "solid_angle",
    ):
        assert np.all(np.isfinite(getattr(T, c))), c
