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


@numba.njit(nogil=True, cache=True)
def decode_byte_offset(src, out):
    """CBF x-CBF_BYTE_OFFSET data (uint8 array) -> all elements of out (1-D int32). Each element is a delta from the
    previous value: one signed byte; or 0x80 then a little-endian int16; 0x8000 then int32; 0x80000000 then int64.
    Returns the bytes consumed, -1 if the data ends early, -2 if a value overflows int32. (As any2zarr's Cython decoder;
    ~25% slower single-threaded on an M4, identical output.)"""
    n = out.shape[0]
    m = src.shape[0]
    i = 0
    j = 0
    value = np.int64(0)
    while i < n:
        if j >= m:
            return -1
        delta = np.int64(np.int8(src[j]))
        j += 1
        if delta == -128:
            if j + 2 > m:
                return -1
            delta = np.int64(np.int16(np.uint16(src[j]) | (np.uint16(src[j + 1]) << np.uint16(8))))
            j += 2
            if delta == -32768:
                if j + 4 > m:
                    return -1
                u = np.uint32(0)
                for b in range(4):
                    u |= np.uint32(src[j + b]) << np.uint32(8 * b)
                delta = np.int64(np.int32(u))
                j += 4
                if delta == -2147483648:
                    if j + 8 > m:
                        return -1
                    u64 = np.uint64(0)
                    for b in range(8):
                        u64 |= np.uint64(src[j + b]) << np.uint64(8 * b)
                    delta = np.int64(u64)
                    j += 8
        value += delta
        if value > 2147483647 or value < -2147483648:
            return -2
        out[i] = np.int32(value)
        i += 1
    return j
