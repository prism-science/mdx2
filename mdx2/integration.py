"""Per-chunk binning of an image series on the (subdivided) Miller-index grid, shared by integrate and reintegrate.

bin_chunk(image_series, sl, miller_index, mask, ndiv) gives the same table as
    image_series[sl].index(miller_index, mask=mask.for_chunk(sl)).bin(count_name="pixels")
(bit for bit), from one compiled pass over the chunk instead of full-size intermediate arrays.

chunk_columns(image_series) groups the chunk slices by detector region (chunk row and column): mdx2's chunking is
module-aligned by default, so a column of chunks along phi covers one module.
"""

import math
import time
from collections import defaultdict

import numexpr as ne
import numpy as np

from mdx2._integration_kernels import bin_chunk_kernel
from mdx2.data import HKLTable
from mdx2.geometry import GaussianPeak, PeakMask, StaticMask, peak_expression
from mdx2.utils import _base_fraction


def chunk_columns(image_series):
    """Chunk slices grouped by (iy, ix) chunk position, each group ordered along phi."""
    cols = defaultdict(list)
    for sl in image_series.chunk_slice_iterator():
        cols[(sl[1].start, sl[2].start)].append(sl)
    return [sorted(v, key=lambda s: s[0].start) for _, v in sorted(cols.items())]


def _peak_parameters(mask):
    """Numbers for the kernel's ellipsoid test, verified to reproduce the mask's stored peak_expression."""
    cached = getattr(mask, "_kernel_parameters", None)
    if cached is not None:
        return cached
    if mask.peak_r0 is None or mask.peak_sigma is None or mask.sigma_cutoff is None:
        raise ValueError("PeakMask without peak_r0 / peak_sigma / sigma_cutoff: cannot be evaluated by the kernel")
    rebuilt = peak_expression(GaussianPeak(mask.peak_r0, mask.peak_sigma), mask.sigma_cutoff)
    if rebuilt != mask.peak_expression:
        raise ValueError("PeakMask: peak_expression does not match peak_r0 / peak_sigma / sigma_cutoff")
    params = (np.linalg.inv(np.asarray(mask.peak_sigma, dtype=float)), np.asarray(mask.peak_r0, dtype=float),
              float(mask.sigma_cutoff) ** 2)  # fmt: skip
    mask._kernel_parameters = params
    return params


def bin_chunk(image_series, sl, miller_index, mask=None, ndiv=(1, 1, 1), timer=None):
    """Per-voxel sums for one chunk: HKLTable (h, k, l; pixels, phi, iy, ix, counts, seconds) or None if empty.

    mask: None, PeakMask (rebuilt here) or StaticMask. timer: optional dict, accumulates seconds per step.
    """
    t = time.perf_counter
    t0 = t()
    ims = image_series[sl]
    data = np.ascontiguousarray(ims.data)
    t1 = t()

    mi = miller_index
    bs = _base_fraction(ims.phi, axis=mi.phi), _base_fraction(ims.iy, axis=mi.iy), _base_fraction(ims.ix, axis=mi.ix)
    order = np.argsort([b.size for b, _ in bs]).astype(np.int64)  # as mdx2.utils._quick_trilinear_g2g
    # grid nodes the chunk touches; trilinear values are convex combinations of them, which bounds h, k, l
    sub = tuple(slice(int(b.min()), int(b.max()) + 2) for b, _ in bs)
    hs, ks, ls = mi.h[sub], mi.k[sub], mi.l[sub]
    Hg, Kg, Lg = (np.ascontiguousarray(np.transpose(a, order)) for a in (hs, ks, ls))
    bsort = [np.ascontiguousarray(bs[o][0] - bs[o][0].min(), dtype=np.int64) for o in order]
    fsort = [np.ascontiguousarray(bs[o][1], dtype=np.float64) for o in order]
    nd = np.asarray(ndiv, dtype=np.int64)
    box_lo = np.array([math.floor(a.min() * n) - 1 for a, n in zip((hs, ks, ls), nd)], dtype=np.int64)
    box_hi = np.array([math.ceil(a.max() * n) + 1 for a, n in zip((hs, ks, ls), nd)], dtype=np.int64)
    box_shape = box_hi - box_lo + 1

    use_peak = use_static = bragg = False
    W, r0, cut2 = np.eye(3), np.zeros(3), 0.0
    allowed, a_lo = np.ones((1, 1, 1), dtype=np.bool_), np.zeros(3, np.int64)
    static = np.zeros((1, 1, 1), dtype=np.bool_)
    if isinstance(mask, PeakMask):
        use_peak, bragg = True, mask.bragg
        W, r0, cut2 = _peak_parameters(mask)
        a_lo = np.array([math.floor(a.min()) - 1 for a in (hs, ks, ls)], dtype=np.int64)
        a_hi = np.array([math.ceil(a.max()) + 1 for a in (hs, ks, ls)], dtype=np.int64)
        Hh, Kk, Ll = np.meshgrid(*(np.arange(lo, hi + 1) for lo, hi in zip(a_lo, a_hi)), indexing="ij")
        allowed = ne.evaluate(mask.reflection_conditions, local_dict={"h": Hh, "k": Kk, "l": Ll})
        allowed = np.ascontiguousarray(np.broadcast_to(allowed, Hh.shape))  # constant conditions are scalars
        extra = mask.chunk_pixels(tuple(slice(*s.indices(n)) for s, n in zip(sl, mask.shape)))
        if extra[0].size:
            if not data.flags.writeable or data is ims.data:
                data = data.copy()
            data[extra] = ims._maskval
    elif isinstance(mask, StaticMask):
        static = np.ascontiguousarray(mask.for_chunk(sl), dtype=np.bool_)
        use_static = True
    elif mask is not None:
        raise TypeError(f"unsupported mask type {type(mask).__name__}")
    t2 = t()

    pixels, s_phi, s_iy, s_ix, s_counts, s_sec = bin_chunk_kernel(
        data, ims._maskval, static, use_static, use_peak, bragg,
        bsort[0], fsort[0], bsort[1], fsort[1], bsort[2], fsort[2], order,
        Hg, Kg, Lg,
        np.asarray(ims.phi, dtype=float), np.asarray(ims.iy, dtype=float), np.asarray(ims.ix, dtype=float),
        np.asarray(ims.exposure_times, dtype=float),
        W, r0, cut2, allowed, a_lo, nd, box_lo, box_shape, bool(order[2] == 2),
    )  # fmt: skip
    t3 = t()
    nz = np.flatnonzero(pixels)
    tab = None
    if nz.size:
        Hb, Kb, Lb = np.unravel_index(nz, tuple(box_shape))
        tab = HKLTable(
            (Hb + box_lo[0]) / nd[0], (Kb + box_lo[1]) / nd[1], (Lb + box_lo[2]) / nd[2], ndiv=tuple(ndiv),
            pixels=pixels[nz], phi=s_phi[nz], iy=s_iy[nz], ix=s_ix[nz], counts=s_counts[nz], seconds=s_sec[nz],
        )  # fmt: skip
    if timer is not None:
        for k, v in (("read", t1 - t0), ("prepare", t2 - t1), ("kernel", t3 - t2), ("table", t() - t3)):
            timer[k] = timer.get(k, 0.0) + v
    return tab


def log_timers(timers, wall, verbose=False):
    """Log seconds per step summed over the tasks (and their share of all task time), against the wall time."""
    from loguru import logger  # noqa: PLC0415

    total = defaultdict(float)
    for t in timers:
        for k, v in t.items():
            total[k] += v
    busy = sum(total.values())
    log = logger.info if verbose else logger.debug
    log("Task time {:.1f} s over {:.1f} s wall ({:.1f} tasks busy on average)", busy, wall, busy / max(wall, 1e-9))
    for k, v in sorted(total.items(), key=lambda kv: -kv[1]):
        log("  {:12s} {:9.1f} s  ({:5.1f}%)", k, v, 100 * v / max(busy, 1e-9))
