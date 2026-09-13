"""
Behaviour of Dataset / Snapshot / CartesianSnapshot.

Written before the Phase 2 refactor to pin down what already works, and extended
with the cases that were silently wrong.
"""

import numpy as np
import pytest

from flowkit import CartesianSnapshot, Dataset, NotAssociatedWithDataset, Snapshot
from tests.conftest import AOA, U_INF


# --------------------------------------------------------------- Dataset.crop

def test_crop_selects_only_cells_in_bounds(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    sub = ds.crop(xmin=0.0, xmax=5.0, ymin=-1.0, ymax=1.0)
    x, y = sub.data["x"].values, sub.data["y"].values
    assert sub.data.sizes["cell"] < ds.data.sizes["cell"]
    assert (x >= 0.0).all() and (x <= 5.0).all()
    assert (y >= -1.0).all() and (y <= 1.0).all()


def test_crop_accepts_partial_bounds(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    sub = ds.crop(xmin=0.0)
    assert (sub.data["x"].values >= 0.0).all()
    assert sub.data.sizes["cell"] > 0


def test_crop_relative_scales_by_length_scale(scattered_case):
    scattered_case.attrs["length_scale"] = 2.0
    ds = Dataset.from_dataset(scattered_case)
    assert ds.length_scale == 2.0
    a = ds.crop_relative(0, 2, -0.5, 0.5)
    b = ds.crop(xmin=0, xmax=4, ymin=-1, ymax=1)
    assert a.data.sizes["cell"] == b.data.sizes["cell"]


def test_crop_relative_without_length_scale_raises(scattered_case):
    del scattered_case.attrs["length_scale"]
    with pytest.raises(ValueError, match="length_scale"):
        Dataset.from_dataset(scattered_case).crop_relative(0, 1, 0, 1)


# ------------------------------------------------------------ rotate / rotated

def test_rotate_maps_freestream_onto_x_axis(uniform_case):
    rot = Dataset.from_dataset(uniform_case).rotate(AOA)
    U = rot.data["U"].values
    assert np.allclose(U[..., 0], 1.0)
    assert np.allclose(U[..., 1], 0.0, atol=1e-12)


def test_rotate_preserves_speed(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    rot = ds.rotate(AOA)
    s0 = np.hypot(ds.data["U"].values[..., 0], ds.data["U"].values[..., 1])
    s1 = np.hypot(rot.data["U"].values[..., 0], rot.data["U"].values[..., 1])
    assert np.abs(s0 - s1).max() < 1e-12


def test_rotate_preserves_distance_from_origin(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    rot = ds.rotate(AOA)
    r0 = np.hypot(ds.data["x"].values, ds.data["y"].values)
    r1 = np.hypot(rot.data["x"].values, rot.data["y"].values)
    assert np.abs(r0 - r1).max() < 1e-12


def test_rotate_records_the_frame(scattered_case):
    rot = Dataset.from_dataset(scattered_case).rotate(AOA, origin=(1.0, 2.0))
    assert rot.data.attrs["frame_angle"] == AOA
    assert tuple(rot.data.attrs["frame_origin"]) == (1.0, 2.0)


def test_rotate_is_invertible(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    back = ds.rotate(AOA).rotate(-AOA)
    assert np.allclose(back.data["x"].values, ds.data["x"].values, atol=1e-12)
    assert np.allclose(back.data["U"].values, ds.data["U"].values, atol=1e-12)


def test_crop_rotated_matches_rotate_then_crop(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    box = dict(xmin=0.0, xmax=6.0, ymin=-1.0, ymax=1.0)
    a = ds.crop_rotated(**box, angle=AOA)
    b = ds.rotate(AOA).crop(**box)
    assert a.data.sizes["cell"] == b.data.sizes["cell"]


def test_crop_rotated_differs_from_axis_aligned(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    box = dict(xmin=0.0, xmax=6.0, ymin=-1.0, ymax=1.0)
    assert ds.crop(**box).data.sizes["cell"] != ds.crop_rotated(**box, angle=AOA).data.sizes["cell"]


# ------------------------------------------------------- the V coordinate (#30)

@pytest.mark.parametrize("op", [
    lambda d: d.crop(xmin=0.0, xmax=5.0),
    lambda d: d.crop_relative(0, 3, -1, 1),
    lambda d: d.crop_rotated(xmin=0.0, xmax=5.0, angle=AOA),
    lambda d: d.crop_rotated_relative(0, 3, -1, 1, angle=AOA),
    lambda d: d.rotate(AOA),
])
def test_volumes_survive_every_transform(scattered_case, op):
    """V is what POD needs; losing it silently is issue #30."""
    out = op(Dataset.from_dataset(scattered_case))
    assert "V" in out.data.coords
    assert out.data.sizes["cell"] == out.data["V"].size
    assert (out.data["V"].values > 0).all()


def test_volumes_survive_netcdf_roundtrip(scattered_case, tmp_path):
    f = tmp_path / "rt.nc"
    Dataset.from_dataset(scattered_case).data.to_netcdf(f)
    assert "V" in Dataset.from_netcdf(f).data.coords


def test_rotate_leaves_volumes_untouched(scattered_case):
    """Rotation is rigid, so volumes must be bit-identical."""
    ds = Dataset.from_dataset(scattered_case)
    assert np.array_equal(ds.rotate(AOA).data["V"].values, ds.data["V"].values)


# -------------------------------------------------------------------- Snapshot

def test_snapshot_exposes_the_selected_time(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    t = float(ds.times[2])
    snap = ds.snapshot(t)
    expect = scattered_case.sel(time=t)
    assert np.allclose(snap.u, expect["U"].values[:, 0])
    assert np.allclose(snap.v, expect["U"].values[:, 1])


def test_snapshot_accepts_a_time_that_is_not_exact(scattered_case):
    """sel() uses nearest, so the index lookup must too -- issue #5."""
    ds = Dataset.from_dataset(scattered_case)
    exact = float(ds.times[2])
    snap = ds.snapshot(exact + 1e-9)
    assert np.allclose(snap.u, ds.snapshot(exact).u)


def test_next_and_previous_walk_the_time_axis(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    snap = ds.snapshot(float(ds.times[2]))
    assert np.allclose(snap.next().u, ds.snapshot(float(ds.times[3])).u)
    assert np.allclose(snap.previous().u, ds.snapshot(float(ds.times[1])).u)


def test_previous_at_the_start_raises_instead_of_wrapping(scattered_case):
    """idx=-1 silently returned the LAST snapshot -- issue #4, the dangerous one."""
    ds = Dataset.from_dataset(scattered_case)
    with pytest.raises(IndexError):
        ds.snapshot(float(ds.times[0])).previous()


def test_next_at_the_end_raises(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    with pytest.raises(IndexError):
        ds.snapshot(float(ds.times[-1])).next()


def test_detached_snapshot_cannot_step(scattered_case):
    snap = Snapshot.from_xarray(scattered_case.isel(time=0))
    with pytest.raises(NotAssociatedWithDataset):
        snap.next()


def test_snapshots_generator_defaults_to_all_times(scattered_case):
    """Default path raised AttributeError on self.time -- issue #3."""
    ds = Dataset.from_dataset(scattered_case)
    assert len(list(ds.snapshots())) == len(ds.times)


# ------------------------------------------------------- Snapshot crop / mask

def test_snapshot_mask_allows_partial_bounds(scattered_case):
    """Omitted bounds used to raise TypeError against None -- issue #7."""
    snap = Dataset.from_dataset(scattered_case).snapshot(0.0)
    m = snap.mask(xmin=0.0)
    assert m.dtype == bool and m.any()
    assert (snap.x[m] >= 0.0).all()


def test_snapshot_crop_takes_bounds_like_dataset(scattered_case):
    snap = Dataset.from_dataset(scattered_case).snapshot(0.0)
    sub = snap.crop(xmin=0.0, xmax=5.0)
    assert sub.x.size < snap.x.size
    assert (sub.x >= 0.0).all() and (sub.x <= 5.0).all()


def test_snapshot_crop_keeps_the_dataset_link(scattered_case):
    """Cropping then stepping through time was impossible -- issue #9."""
    ds = Dataset.from_dataset(scattered_case)
    sub = ds.snapshot(float(ds.times[1])).crop(xmin=0.0, xmax=5.0)
    nxt = sub.next()
    assert nxt.x.size == sub.x.size


def test_mask_rotated_agrees_with_rotate_then_mask(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    snap = ds.snapshot(0.0)
    a = snap.mask_rotated(xmin=0.0, xmax=5.0, angle=AOA).sum()
    b = ds.rotate(AOA).snapshot(0.0).mask(xmin=0.0, xmax=5.0).sum()
    assert a == b


# ---------------------------------------------------------- Cartesian gridding

def test_default_element_size_is_two_dimensional(gridded_case):
    """
    (xmax-xmin)/n is a 1-D formula and produced grids ~1e5 too fine -- issue #6.
    A sane default is near sqrt(area/n).
    """
    snap = Dataset.from_dataset(gridded_case).snapshot(0.0)
    gx, gy = snap.cartesian_grid()
    expected = np.sqrt((snap.xmax - snap.xmin) * (snap.ymax - snap.ymin) / snap.x.size)
    assert 0.25 * expected < (snap.xmax - snap.xmin) / gx.size < 4 * expected
    assert gx.size * gy.size < 20 * snap.x.size


def test_grid_reaches_the_upper_bound(gridded_case):
    """arange() dropped the last row/column -- issue #8."""
    snap = Dataset.from_dataset(gridded_case).snapshot(0.0)
    gx, gy = snap.cartesian_grid(element_size=0.1)
    assert gx[-1] >= snap.xmax - 1e-9
    assert gy[-1] >= snap.ymax - 1e-9


def test_interpolation_reproduces_a_linear_field(gridded_case):
    """u = 1 + x/2 is linear, so linear interpolation must be exact."""
    snap = Dataset.from_dataset(gridded_case).snapshot(0.0)
    f = snap.interpolate_on_grid(element_size=0.1)
    X, Y = np.meshgrid(f.x, f.y)
    ok = ~np.isnan(f.u)
    assert np.abs(f.u[ok] - (1.0 + 0.5 * X[ok])).max() < 1e-10
    assert np.abs(f.v[ok] - (-0.25 * Y[ok])).max() < 1e-10


def test_sample_matches_the_analytic_field(gridded_case):
    snap = Dataset.from_dataset(gridded_case).snapshot(0.0)
    f = snap.interpolate_on_grid(element_size=0.1)
    u, v, p = f.sample(np.array([1.0, 2.0]), np.array([0.0, 0.5]))
    assert np.allclose(u, [1.5, 2.0], atol=1e-10)
    assert np.allclose(v, [0.0, -0.125], atol=1e-10)


def test_sample_returns_nan_outside_the_grid(gridded_case):
    snap = Dataset.from_dataset(gridded_case).snapshot(0.0)
    f = snap.interpolate_on_grid(element_size=0.1)
    u, v, p = f.sample(np.array([99.0]), np.array([99.0]))
    assert np.isnan(u).all()


def test_cartesian_snapshot_rejects_mismatched_shapes():
    x, y = np.linspace(0, 1, 5), np.linspace(0, 1, 4)
    good = np.zeros((4, 5))
    with pytest.raises(ValueError):
        CartesianSnapshot(x, y, good, good, np.zeros((5, 4)))


# ------------------------------------------------------------------- ergonomics

def test_objects_have_useful_repr(scattered_case):
    ds = Dataset.from_dataset(scattered_case)
    snap = ds.snapshot(0.0)
    for obj, word in ((ds, "Dataset"), (snap, "Snapshot")):
        r = repr(obj)
        assert word in r and "0x" not in r
