"""
Import Bragg intensities integrated by DIALS (summation) on the mdx2 intensity scale
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from dxtbx.model.experiment_list import ExperimentList
from loguru import logger
from simple_parsing import field

import mdx2.dxtbx_machinery as dm
from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.data import HKLTable
from mdx2.dials_refl import FLAGS, read_refl
from mdx2.geometry import Symmetry
from mdx2.io import loadobj, saveobj

COLUMNS = [
    "id", "panel", "flags", "miller_index", "xyzcal.px", "xyzcal.mm", "d", "lp", "qe", "partiality", "zeta", "bbox",
    "intensity.sum.value", "intensity.sum.variance", "background.sum.value", "background.sum.variance",
    "background.mean", "num_pixels.foreground", "num_pixels.background_used",
]  # fmt: skip


@dataclass
class Parameters:
    """Options for importing DIALS-integrated Bragg intensities"""

    refl: str = field(positional=True)  # DIALS reflection file, such as integrated.refl
    expt: str = field(positional=True)  # matching DIALS experiments file, such as integrated.expt
    experiment: Optional[int] = None  # index of the experiment to import (required if the file has several)
    background: Optional[str] = None
    """NeXus file with the binned experimental background (binned_image_series, as for mdx2.correct); adds
    background_run_intensity and background_diffuse"""
    outfile: str = "bragg.nxs"  # name of the output NeXus file
    logfile: str = "mdx2.import_bragg.log"  # name of the log file


def symmetry_from_experiment(expt):
    E = dm.Experiment(expt)
    return Symmetry(
        space_group_number=E.space_group_number,
        space_group_symbol=E.space_group_symbol,
        crystal_system=E.crystal_system,
        space_group_operators=E.space_group_operators,
        laue_group_number=E.laue_group_number,
        laue_group_symbol=E.laue_group_symbol,
        laue_group_operators=E.laue_group_operators,
        reciprocal_space_asu=E.reciprocal_space_asu,
        reflection_conditions=E.reflection_conditions,
    )


def corrections_at(expt, panels, ix, iy):
    """mdx2's geometric correction factors at pixel positions (per panel), as in Corrections.from_expt."""
    n = ix.size
    out = {k: np.empty(n) for k in ("solid_angle", "efficiency", "attenuation", "polarization", "inverse_lorentz")}
    for p in np.unique(panels):
        sel = panels == p
        panel = expt.detector[int(p)]
        xyz = dm.pixels_to_lab(panel, ix[sel], iy[sel])
        d = np.sqrt(np.sum(xyz * xyz, axis=1))
        cosw = xyz @ np.array(panel.get_normal()) / d
        out["solid_angle"][sel] = dm.calc_solid_angle(panel, cosw, d)
        out["efficiency"][sel] = dm.calc_efficiency(panel, cosw)
        out["attenuation"][sel] = dm.calc_attenuation(expt.beam, d)
        out["polarization"][sel] = dm.calc_polarization(expt.beam, xyz, d)
        out["inverse_lorentz"][sel] = dm.calc_inverse_lorentz(expt.beam, expt.goniometer, xyz, d)
    return out


def run_import_bragg(params):
    """Run the import script"""
    elist = ExperimentList.from_file(params.expt, check_format=False)
    if params.experiment is None and len(elist) > 1:
        raise ValueError(f"{params.expt} has {len(elist)} experiments: choose one with --experiment")
    k = params.experiment or 0
    expt = elist[k]
    scan = expt.scan

    logger.info("Reading reflections from {}...", params.refl)
    R = read_refl(params.refl, COLUMNS)
    ids = [i for i, uid in R.identifiers.items() if uid == expt.identifier]
    if len(ids) != 1:
        raise ValueError(f"experiment {k} (identifier {expt.identifier}) not found in {params.refl}")
    R = R.select(R["id"] == ids[0])
    n_all = len(R)
    R = R.select(R.has_flag("integrated_sum"))
    logger.info("Experiment {}: {} reflections, {} integrated by summation", k, n_all, len(R))
    if len(R) == 0:
        raise ValueError("no reflections integrated by summation")

    ix, iy, z = R["xyzcal.px"].T
    phi = np.degrees(R["xyzcal.mm"][:, 2])
    frame = np.clip(np.floor(z).astype(int) - scan.get_array_range()[0], 0, scan.get_num_images() - 1)
    exposure = np.asarray(scan.get_exposure_times(), dtype=float)[frame]
    dphi = np.radians(scan.get_oscillation()[1])
    wavelength = expt.beam.get_wavelength()
    crystal = expt.crystal
    if crystal.num_scan_points > 0:
        dets = np.array(
            [np.linalg.det(np.reshape(crystal.get_A_at_scan_point(i), (3, 3))) for i in range(crystal.num_scan_points)]
        )
        det_A = dets[np.clip(frame, 0, crystal.num_scan_points - 1)]
        logger.info("Scan-varying crystal model: reciprocal cell volume per image")
    else:
        det_A = np.linalg.det(np.reshape(crystal.get_A(), (3, 3)))

    logger.info("Computing correction factors at the predicted positions...")
    C = corrections_at(expt, R["panel"], ix, iy)
    lp, qe = R["lp"], R["qe"]
    for name, ours, theirs in (
        ("Linv / polarization", C["inverse_lorentz"] / C["polarization"], lp),
        ("efficiency", C["efficiency"], qe),
    ):
        dev = float(np.max(np.abs(ours / theirs - 1)))
        (logger.warning if dev > 1e-3 else logger.info)("{} vs DIALS: max relative deviation {:.2e}", name, dev)

    # Lorentz-corrected integrated intensity per reciprocal unit cell (the mdx2.correct convention with Lorentz)
    iscale = dphi / (exposure * wavelength**3 * det_A)
    gain = iscale * lp / (qe * C["attenuation"])
    I_sum, V_sum = R["intensity.sum.value"], R["intensity.sum.variance"]

    # DIALS local background in mdx2 diffuse units (corrected count rate per steradian)
    denom = C["solid_angle"] * C["attenuation"] * C["efficiency"] * C["polarization"]
    npix = np.maximum(R["num_pixels.foreground"], 1)
    bkg_rate = R["background.mean"] / exposure
    bkg_rate_err = np.sqrt(np.maximum(R["background.sum.variance"], 0)) / npix / exposure

    cols = dict(
        intensity=gain * I_sum,
        intensity_error=gain * np.sqrt(np.maximum(V_sum, 0)),
        phi=phi,
        ix=ix,
        iy=iy,
        s=1.0 / R["d"],
        background_intensity=bkg_rate / denom,
        background_intensity_error=bkg_rate_err / denom,
        intensity_sum=I_sum,
        intensity_sum_variance=V_sum,
        background_sum=R["background.sum.value"],
        background_mean=R["background.mean"],
        pixels_foreground=R["num_pixels.foreground"],
        pixels_background=R["num_pixels.background_used"],
        lp=lp,
        qe=qe,
        partiality=R["partiality"],
        zeta=R["zeta"],
        frames=R["bbox"][:, 5] - R["bbox"][:, 4],
        dials_flags=R["flags"].astype(np.int64),
        **C,
    )
    if params.background is not None:
        run = loadobj(params.background, "binned_image_series").interpolate(phi, iy, ix)
        cols["background_run_intensity"] = run / denom
        cols["background_diffuse"] = (bkg_rate - run) / denom
        logger.info("Background run: {:.2%} of positions outside its map", float(np.mean(np.isnan(run))))

    h, k_, l = (R["miller_index"][:, j].astype(float) for j in range(3))
    keep64 = {"intensity_sum", "intensity_sum_variance", "background_sum"}
    cols = {c: (v.astype(np.float32) if v.dtype == np.float64 and c not in keep64 else v) for c, v in cols.items()}
    T = HKLTable(h, k_, l, ndiv=(1, 1, 1), **cols).to_asu(symmetry_from_experiment(expt))
    T.h, T.k, T.l = (np.asarray(x, dtype=np.float32) for x in (T.h, T.k, T.l))

    full = T.partiality > 0.99
    near_axis = T.inverse_lorentz < 0.05
    fpf = (T.dials_flags & FLAGS["failed_during_profile_fitting"]) != 0
    logger.info("Prefactor dphi V_cell / (t lambda^3): {:.6g} (median)", float(np.median(iscale)))
    logger.info(
        "Full (partiality > 0.99): {:.1%}; near the rotation axis (inverse_lorentz < 0.05): {:.2%}; "
        "failed_during_profile_fitting: {:.2%}",
        float(np.mean(full)),
        float(np.mean(near_axis)),
        float(np.mean(fpf)),
    )
    logger.info("Saving {} reflections to {}...", len(T), params.outfile)
    saveobj(T, params.outfile, name="hkl_table", append=False)
    logger.info("Import completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_import_bragg))

if __name__ == "__main__":
    run()
