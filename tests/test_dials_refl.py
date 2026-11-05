"""DIALS reflection-table reader (mdx2.dials_refl), without DIALS.

Synthetic files always; against DIALS itself on a real file when MDX2_TEST_REFL=<integrated.refl> is set and DIALS
is importable.
"""

import os

import msgpack
import numpy as np
import pytest

from mdx2.dials_refl import FLAGS, read_refl


def write_refl(path, cols, identifiers=None, version=1, header="dials::af::reflection_table"):
    n = len(next(iter(cols.values()))[1])
    data = {name: [tname, [n, np.ascontiguousarray(a).tobytes()]] for name, (tname, a) in cols.items()}
    doc = [header, version, {"identifiers": identifiers or {0: "abc"}, "nrows": n, "data": data}]
    with open(path, "wb") as f:
        f.write(msgpack.packb(doc, use_bin_type=True))


def synthetic(n=7, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "d": ("double", rng.random(n)),
        "id": ("int", np.arange(n, dtype="<i4") % 2),
        "flags": (
            "std::size_t",
            np.array([FLAGS["integrated_sum"] | FLAGS["indexed"], FLAGS["indexed"]] * n, "<u8")[:n],
        ),
        "entering": ("bool", rng.random(n) > 0.5),
        "xyzcal.px": ("vec3<double>", rng.random((n, 3))),
        "xy": ("vec2<double>", rng.random((n, 2))),
        "miller_index": ("cctbx::miller::index<>", rng.integers(-20, 20, (n, 3)).astype("<i4")),
        "bbox": ("int6", rng.integers(0, 100, (n, 6)).astype("<i4")),
        "m": ("mat3<double>", rng.random((n, 9))),
    }


def test_round_trip(tmp_path):
    cols = synthetic()
    cols["shoebox"] = ("Shoebox<>", np.zeros(7, dtype="<u1"))  # unsupported: skipped unless requested
    write_refl(tmp_path / "t.refl", cols, identifiers={0: "x", 1: "y"})
    R = read_refl(tmp_path / "t.refl")
    assert len(R) == 7 and R.identifiers == {0: "x", 1: "y"}
    assert R.skipped == {"shoebox": "Shoebox<>"} and "shoebox" not in R
    for name, (_, a) in cols.items():
        if name != "shoebox":
            np.testing.assert_array_equal(R[name], a, err_msg=name)
    assert R["xyzcal.px"].shape == (7, 3) and R["bbox"].shape == (7, 6) and R["m"].shape == (7, 9)
    s = R.select(R.has_flag("integrated_sum"))
    assert len(s) == 4 and np.all(s["id"] == 0)


def test_selected_columns_and_errors(tmp_path):
    cols = synthetic()
    cols["shoebox"] = ("Shoebox<>", np.zeros(7, dtype="<u1"))
    write_refl(tmp_path / "t.refl", cols)
    R = read_refl(tmp_path / "t.refl", ["d", "miller_index"])
    assert sorted(R.keys()) == ["d", "miller_index"]
    with pytest.raises(KeyError):
        read_refl(tmp_path / "t.refl", ["d", "nonexistent"])
    with pytest.raises(ValueError, match="unsupported type"):
        read_refl(tmp_path / "t.refl", ["shoebox"])
    write_refl(tmp_path / "bad.refl", synthetic(), header="something else")
    with pytest.raises(ValueError, match="not a DIALS reflection table"):
        read_refl(tmp_path / "bad.refl")
    write_refl(tmp_path / "v2.refl", synthetic(), version=2)
    with pytest.raises(ValueError, match="version"):
        read_refl(tmp_path / "v2.refl")


REFL = os.environ.get("MDX2_TEST_REFL")


@pytest.mark.skipif(not REFL, reason="set MDX2_TEST_REFL=<integrated.refl> to compare with DIALS")
def test_matches_dials():
    flex = pytest.importorskip("dials.array_family.flex")
    ref = flex.reflection_table.from_file(REFL)
    R = read_refl(REFL)
    assert len(R) == len(ref)
    assert R.identifiers == dict(ref.experiment_identifiers())
    for name in R.keys():
        col = ref[name]
        a = np.array(col.as_vec3_double() if name == "miller_index" else col)
        np.testing.assert_array_equal(R[name], a.reshape(R[name].shape).astype(R[name].dtype), err_msg=name)
    for name, bit in FLAGS.items():
        assert int(getattr(flex.reflection_table.flags, name)) == bit, name
