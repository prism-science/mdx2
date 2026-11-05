"""Compiled single-pass kernels for mdx2.import_data: frames to / from the uint16 assembly buffer.

Codes in the buffer: 0..65533 are counts, MASKED16 (65535) is a masked pixel (-1 in the datastore), OVERFLOW16 (65534)
means the value is kept in a separate overflow list.
"""

import numba
import numpy as np

MASKED16 = 65535
OVERFLOW16 = 65534


@numba.njit(nogil=True, cache=True)
def pack_frame_kernel(frame, buf):
    """int32 frame (masked = -1) -> uint16 codes in buf; returns the number of OVERFLOW16 pixels."""
    n = 0
    ny, nx = frame.shape
    for y in range(ny):
        for x in range(nx):
            v = frame[y, x]
            if 0 <= v < OVERFLOW16:
                buf[y, x] = v
            elif v == -1:
                buf[y, x] = MASKED16
            else:
                buf[y, x] = OVERFLOW16
                n += 1
    return n


@numba.njit(nogil=True, cache=True)
def pack_raw_kernel(raw, lo, hi, valid, use_valid, buf):
    """Detector frame (any integer type) -> uint16 codes, masking values outside [lo, hi] and where valid is False
    (if use_valid), as the dxtbx trusted-range + static mask; returns the number of OVERFLOW16 pixels."""
    n = 0
    ny, nx = raw.shape
    for y in range(ny):
        for x in range(nx):
            v = raw[y, x]
            if v < lo or v > hi or (use_valid and not valid[y, x]):
                buf[y, x] = MASKED16
            elif v < OVERFLOW16:
                buf[y, x] = v
            else:
                buf[y, x] = OVERFLOW16
                n += 1
    return n


@numba.njit(nogil=True, cache=True)
def unpack_kernel(codes, out):
    """uint16 codes -> int32 (MASKED16 -> -1; OVERFLOW16 left as 65534, to be patched from the overflow list)."""
    nz, ny, nx = codes.shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                c = codes[z, y, x]
                out[z, y, x] = -1 if c == MASKED16 else np.int32(c)


def pack_frame(frame, buf):
    """int32 frame (masked = -1) -> uint16 codes in buf. Returns the overflow pixels (y, x, int32 values) or None."""
    if pack_frame_kernel(frame, buf):
        y, x = np.nonzero(buf == OVERFLOW16)
        return y, x, frame[y, x].astype(np.int32)
    return None


_ALL_VALID = np.ones((1, 1), dtype=np.bool_)


def pack_raw(raw, lo, hi, static_mask, buf):
    """Detector frame -> uint16 codes in buf, masked outside [lo, hi] and where static_mask is False (if not None).
    Returns the overflow pixels (y, x, int32 values) or None."""
    valid = static_mask if static_mask is not None else _ALL_VALID
    if pack_raw_kernel(raw, lo, hi, valid, static_mask is not None, buf):
        y, x = np.nonzero(buf == OVERFLOW16)
        return y, x, raw[y, x].astype(np.int32)
    return None
