"""Read DIALS reflection tables (.refl) without DIALS.

A .refl file is a msgpack document ``["dials::af::reflection_table", version, {"identifiers": {id: uuid},
"nrows": n, "data": {column: [type, [n, raw bytes]]}}]``; array columns are stored as raw little-endian buffers.
This reader decodes the plain array types (numbers, vectors, Miller indices, bounding boxes) into numpy arrays and
skips structured ones (shoeboxes, strings) unless requested explicitly.
"""

import msgpack
import numpy as np

# element dtype and number of components per row
_TYPES = {
    "double": ("<f8", 1),
    "int": ("<i4", 1),
    "std::size_t": ("<u8", 1),
    "bool": ("|b1", 1),
    "vec2<double>": ("<f8", 2),
    "vec3<double>": ("<f8", 3),
    "mat3<double>": ("<f8", 9),
    "int6": ("<i4", 6),
    "cctbx::miller::index<>": ("<i4", 3),
}

# DIALS reflection_table.flags (dials.array_family.flex), checked against DIALS in the tests
FLAGS = {
    "predicted": 1,
    "observed": 2,
    "indexed": 4,
    "used_in_refinement": 8,
    "strong": 32,
    "reference_spot": 64,
    "dont_integrate": 128,
    "integrated_sum": 256,
    "integrated_prf": 512,
    "overloaded": 1024,
    "overlapped_bg": 2048,
    "overlapped_fg": 4096,
    "in_powder_ring": 8192,
    "foreground_includes_bad_pixels": 16384,
    "background_includes_bad_pixels": 32768,
    "used_in_modelling": 65536,
    "centroid_outlier": 131072,
    "failed_during_background_modelling": 262144,
    "failed_during_summation": 524288,
    "failed_during_profile_fitting": 1048576,
    "bad_reference": 2097152,
    "user_excluded_in_scaling": 4194304,
    "outlier_in_scaling": 8388608,
    "excluded_for_scaling": 16777216,
    "excluded_for_refinement": 33554432,
    "scaled": 67108864,
}


class ReflectionTable:
    """Columns of a DIALS reflection table as numpy arrays (vector columns have shape (n, k))."""

    def __init__(self, columns, identifiers, nrows, skipped=()):
        self.columns = columns
        self.identifiers = identifiers  # {experiment id: uuid}
        self.nrows = nrows
        self.skipped = dict(skipped)  # column -> type name, not decoded

    def __getitem__(self, key):
        return self.columns[key]

    def __contains__(self, key):
        return key in self.columns

    def __len__(self):
        return self.nrows

    def keys(self):
        return self.columns.keys()

    def has_flag(self, name):
        """Boolean array: rows with the named DIALS flag set (all bits of a composite flag)."""
        bit = FLAGS[name]
        return (self.columns["flags"] & bit) == bit

    def select(self, mask):
        return ReflectionTable(
            {k: v[mask] for k, v in self.columns.items()}, self.identifiers, int(np.count_nonzero(mask)), self.skipped
        )


def read_refl(path, columns=None):
    """Read a DIALS .refl file. columns: names to decode (default: every column of a supported type)."""
    with open(path, "rb") as f:
        doc = msgpack.unpackb(f.read(), strict_map_key=False, raw=False)
    if not (isinstance(doc, list) and len(doc) == 3 and doc[0] == "dials::af::reflection_table"):
        raise ValueError(f"{path} is not a DIALS reflection table (msgpack format)")
    if doc[1] != 1:
        raise ValueError(f"{path}: unsupported reflection table format version {doc[1]}")
    body = doc[2]
    nrows = int(body["nrows"])
    out, skipped = {}, {}
    wanted = None if columns is None else set(columns)
    for name, (tname, payload) in body["data"].items():
        if wanted is not None and name not in wanted:
            continue
        if tname not in _TYPES:
            if wanted is not None:
                raise ValueError(f"{path}: column {name!r} has unsupported type {tname!r}")
            skipped[name] = tname
            continue
        dtype, k = _TYPES[tname]
        n, buf = payload
        if n != nrows:
            raise ValueError(f"{path}: column {name!r} has {n} rows, table has {nrows}")
        a = np.frombuffer(buf, dtype=dtype)
        if a.size != n * k:
            raise ValueError(f"{path}: column {name!r}: {a.size} values for {n} rows of {tname}")
        out[name] = a.reshape(n, k) if k > 1 else a
    if wanted is not None and (missing := wanted - set(out)):
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    return ReflectionTable(out, dict(body.get("identifiers", {})), nrows, skipped)
