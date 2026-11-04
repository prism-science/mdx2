"""mdx2.merge (compiled kernels, mdx2.scaling) must reproduce the merge of mdx2 <= 1.0.5 (mdx2.scaling_reference).

Synthetic two-batch data and scaling models with random control points, written to NeXus files; the oracle below is
the previous run_merge (reference classes, vectorized weighted random split).
"""

import numba
import numpy as np
import pytest
from nexusformat.nexus import nxload

import mdx2.command_line.merge as merge_cli
from mdx2.command_line.merge import Parameters, run_merge
from mdx2.data import HKLTable
from mdx2.io import loadobj, saveobj
from mdx2.scaling import BatchModelRefiner, ScaledData
from mdx2.scaling_reference import BatchModelRefiner as RefBatchModelRefiner
from mdx2.scaling_reference import ScaledData as RefScaledData

from .test_scaling_engines import synthetic

NAMES = ("scaling", "absorption", "offset", "detector")


def write_inputs(tmp_path, seed=1):
    D = synthetic(n_groups=2000, mult=6)
    rng = np.random.default_rng(seed)
    G = D["ih"].max() + 1
    # (h, k, l) per group on a 2x2x2 subdivision grid; table rows grouped by batch
    hkl = np.stack(np.unravel_index(rng.permutation(16 * 16 * 16)[:G], (16, 16, 16))) / 2.0 - 4.0
    D["I"][rng.random(D["I"].size) < 0.01] = np.nan  # a few invalid observations
    hkl_files, scale_files = [], []
    for b in (0, 1):
        sel = D["batch"] == b
        cols = {k: D[k][sel] for k in ("phi", "s", "ix", "iy")}
        tab = HKLTable(
            *(c[D["ih"][sel]] for c in hkl),
            ndiv=(2, 2, 2),
            op=np.zeros(sel.sum(), dtype=int),
            intensity=D["I"][sel],
            intensity_error=D["sigma"][sel],
            **cols,
        )
        fn = str(tmp_path / f"corrected_{b}.nxs")
        saveobj(tab, fn, name="hkl_table", append=False)
        hkl_files.append(fn)
    # models with the grids mdx2.scale would choose, random control points
    S = ScaledData(D["I"], D["sigma"], D["ih"], phi=D["phi"], ix=D["ix"], iy=D["iy"], s=D["s"], batch=D["batch"])
    MR = BatchModelRefiner(S)
    MR.add_scaling_models(dphi=1.0)
    MR.add_offset_models(dphi=5.0, ns=11)
    MR.add_absorption_models(dphi=10.0, nix=5, niy=5)
    MR.add_detector_model(nix=20, niy=20)
    for b, br in enumerate(MR.batch_refiners):
        fn = str(tmp_path / f"scales_{b}.nxs")
        for j, name in enumerate(NAMES):
            m = getattr(br, name).model.copy()
            n = m.u.size
            u = 20 * rng.random(n) if name == "offset" else 1 + 0.2 * (rng.random(n) - 0.5)
            m.set_u(u.reshape(m.u.shape))
            saveobj(m, fn, name=f"{name}_model", append=j > 0)
        scale_files.append(fn)
    return hkl_files, scale_files


def reference_merge(params):
    """run_merge of mdx2 1.0.5 (columns only)."""
    tabs = []
    for n, fn in enumerate(params.hkl):
        t = loadobj(fn, "hkl_table")
        t.batch = n * np.ones_like(t.op)
        tabs.append(t)
    hkl = HKLTable.concatenate(tabs)
    (h, k, l), index_map, _ = hkl.unique()
    S = RefScaledData(
        hkl.intensity, hkl.intensity_error, index_map, phi=hkl.phi, s=hkl.s, ix=hkl.ix, iy=hkl.iy, batch=hkl.batch
    )
    MR = RefBatchModelRefiner(S)
    use = dict(scaling=params.scaling, absorption=params.absorption, offset=params.offset, detector=params.detector)
    for fn, refiner in zip(params.scale, MR.batch_refiners):
        keys = nxload(fn).entry.keys()
        for name in NAMES:
            if use[name] and f"{name}_model" in keys:
                getattr(refiner, name).model = loadobj(fn, f"{name}_model")
    MR.apply()
    Im, sigmam, counts = MR.data.merge()
    if params.outlier is not None:
        MR.data.mask_outliers(Im, params.outlier)
        Im, sigmam, counts = MR.data.merge()
    cols = dict(intensity=Im.filled(np.nan), intensity_error=sigmam.filled(np.nan), count=counts)
    if params.split is not None:
        # float64: 1.0.5 used float32 weights, whose dataset-wide cumulative sum misassigns ~4% of observations
        w = (1 / MR.data.sigma**2).filled(0).astype(np.float64) if params.split == "weightedRandomHalf" else None
        w = np.ones(len(index_map)) if w is None else w
        isgrp1 = _wrs(index_map, w)
        for j, g in enumerate([isgrp1, ~isgrp1]):
            G = MR.data.copy()
            G.mask = G.mask | ~g
            Imj, sigmamj, countsj = G.merge()
            cols[f"group_{j}_intensity"] = Imj.filled(np.nan)
            cols[f"group_{j}_intensity_error"] = sigmamj.filled(np.nan)
            cols[f"group_{j}_count"] = countsj
    return (h, k, l), cols


def _wrs(index_map, w):
    """mdx2 1.0.5 weighted random split."""
    index_jitter = np.random.random_sample(index_map.shape) * 0.2
    sort_order = np.argsort(index_map + index_jitter)
    sorted_ind_map = index_map[sort_order]
    sorted_w = w[sort_order]
    breakpoints = [0] + list(np.nonzero(np.diff(sorted_ind_map))[0] + 1) + [len(sorted_ind_map)]
    starts = np.array(breakpoints[:-1])
    stops = np.array(breakpoints[1:])
    isgrp1 = np.full_like(index_map, False, dtype=bool)
    flips = np.random.randint(2, size=len(starts))
    cs = np.cumsum(sorted_w)
    dcs = cs - cs[starts][sorted_ind_map]
    isincl = dcs > 0.5 * dcs[stops - 1][sorted_ind_map]
    isincl = np.logical_xor(flips[sorted_ind_map], isincl)
    isgrp1[sort_order[isincl]] = True
    return isgrp1


@pytest.mark.parametrize(
    "opts",
    [
        dict(),
        dict(outlier=3.0, split="weightedRandomHalf"),
        dict(outlier=2.0, split="randomHalf", offset=False, detector=False),
    ],
)
@pytest.mark.parametrize("nthreads", [1, None])
def test_merge_matches_reference(tmp_path, opts, nthreads):
    numba.set_num_threads(nthreads or numba.config.NUMBA_NUM_THREADS)
    hkl_files, scale_files = write_inputs(tmp_path)
    outfile = str(tmp_path / "merged.nxs")
    params = Parameters(
        hkl=hkl_files, scale=scale_files, outfile=outfile, nproc=nthreads or -1, outlier_method="legacy", **opts
    )
    np.random.seed(7)
    (h, k, l), ref = reference_merge(params)
    np.random.seed(7)
    run_merge(params)
    out = loadobj(params.outfile, "hkl_table")
    for a, b in zip((out.h, out.k, out.l), (h, k, l)):
        np.testing.assert_array_equal(a, b)
    assert set(out._data_keys) == set(ref)
    # The weighted split compares cumulative weights with half the group total. mdx2 <= 1.0.5 kept scale factors
    # in float32, so near-ties can fall the other way (here ~1 group in 2000, margin ~1e-7); unweighted: exact.
    same = np.ones(len(h), dtype=bool)
    if "group_0_count" in ref:
        same = ref["group_0_count"] == out.group_0_count
        assert (~same).sum() <= (0.002 * len(h) if opts.get("split") == "weightedRandomHalf" else 0)
    for key, ref_col in ref.items():
        got, want = getattr(out, key), ref_col
        if key.startswith("group_"):
            got, want = got[same], want[same]
        if key.endswith("count"):
            np.testing.assert_array_equal(got, want, err_msg=key)
        else:
            np.testing.assert_allclose(got, want.astype(np.float32), rtol=1e-6, equal_nan=True, err_msg=key)


def test_merge_two_sided_outliers(tmp_path, monkeypatch):
    """Default method: repeated rounds until none, capped (a cap of 1 stops after one round)."""
    hkl_files, scale_files = write_inputs(tmp_path)
    counts = {}
    for cap in (100, 1):
        monkeypatch.setattr(merge_cli, "MAX_OUTLIER_ROUNDS", cap)
        out = str(tmp_path / f"merged_{cap}.nxs")
        run_merge(Parameters(hkl=hkl_files, scale=scale_files, outfile=out, outlier=3.0))
        counts[cap] = loadobj(out, "hkl_table").count.sum()
    assert counts[100] < counts[1]  # more rounds, more outliers removed
