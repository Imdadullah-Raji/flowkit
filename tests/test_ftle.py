"""
FTLE validation.

Two levels of check. A uniform strain field has a closed-form answer -- FTLE is
exactly the strain rate everywhere -- which pins the maths to machine precision.
The double gyre then checks the physics: its gyre boundary sits at x=1 in the
steady case, and the ridge must land there.

Both exclude the domain walls, where np.gradient falls back to one-sided
differences and the walls are themselves stagnation lines.
"""

import numpy as np
import pytest

from flowkit import Dataset, double_gyre, flow_map, ftle

INTERIOR = 0.15          # keep this far away from the walls


class Strain:
    """u = (a x, -a y). The flow map is diag(e^aT, e^-aT), so FTLE == a exactly."""

    time_dependent = True

    def __init__(self, a):
        self.a = a

    def sample(self, x, y, t):
        return self.a * np.asarray(x, float), -self.a * np.asarray(y, float)


@pytest.mark.parametrize("a,T", [(0.3, 2.0), (0.5, 1.0), (0.2, 5.0)])
def test_uniform_strain_gives_the_exact_strain_rate(a, T):
    """The sharpest check available: an analytic FTLE, uniform over the grid."""
    axis = np.linspace(-1.0, 1.0, 21)
    f = ftle(Strain(a), axis, axis, t0=0.0, T=T, dt=0.01).ftle
    assert np.abs(f - a).max() < 1e-10


def test_backward_strain_also_recovers_the_rate():
    axis = np.linspace(-1.0, 1.0, 21)
    f = ftle(Strain(0.3), axis, axis, t0=0.0, T=-2.0, dt=0.01).ftle
    assert np.abs(f - 0.3).max() < 1e-10


def _interior_profile(res):
    """Mean FTLE per column, with the wall columns masked out."""
    keep = (res.x > INTERIOR) & (res.x < res.x.max() - INTERIOR)
    return res.x[keep], np.nanmean(res.ftle, axis=0)[keep]


@pytest.fixture(scope="module")
def steady_ftle():
    gyre = double_gyre(epsilon=0.0)
    x = np.linspace(0.02, 1.98, 100)
    y = np.linspace(0.02, 0.98, 50)
    return ftle(gyre, x, y, t0=0.0, T=15.0, dt=0.1)


def test_ridge_sits_on_the_gyre_boundary(steady_ftle):
    """The separatrix of the steady double gyre is exactly x = 1."""
    x, col = _interior_profile(steady_ftle)
    assert abs(x[np.nanargmax(col)] - 1.0) < 0.05


def test_ridge_is_a_local_maximum(steady_ftle):
    """The boundary must stand out from the gyre interiors on either side."""
    x, col = _interior_profile(steady_ftle)
    near = np.abs(x - 1.0) < 0.05
    flank = (np.abs(x - 1.0) > 0.3) & (np.abs(x - 1.0) < 0.7)
    assert np.nanmax(col[near]) > np.nanmax(col[flank])


def test_result_shape_matches_the_seeding_grid(steady_ftle):
    assert steady_ftle.ftle.shape == (steady_ftle.y.size, steady_ftle.x.size)
    assert steady_ftle.flowmap_x.shape == steady_ftle.ftle.shape


def test_flow_map_keeps_the_grid_shape():
    """Escaped particles are NaN in place, never dropped."""
    gyre = double_gyre(epsilon=0.0)
    x, y = np.linspace(0.05, 1.95, 20), np.linspace(0.05, 0.95, 10)
    fx, fy = flow_map(gyre, x, y, t0=0.0, T=5.0, dt=0.1)
    assert fx.shape == (10, 20)


def test_ftle_is_mostly_finite(steady_ftle):
    assert np.isfinite(steady_ftle.ftle).mean() > 0.9
    assert steady_ftle.escaped_fraction < 0.1


def test_unsteady_gyre_ridge_leaves_the_steady_position(steady_ftle):
    """
    With epsilon=0.25 the gyres breathe, so the ridge meanders off x=1 -- the
    behaviour that makes the unsteady case the standard FTLE demonstration.
    """
    x, y = np.linspace(0.02, 1.98, 100), np.linspace(0.02, 0.98, 50)
    res = ftle(double_gyre(), x, y, t0=0.0, T=15.0, dt=0.1)
    xs, col = _interior_profile(res)
    moved = xs[np.nanargmax(col)]
    assert abs(moved - 1.0) > 0.05
    assert 0.5 < moved < 1.5          # meanders, does not fly off


def test_ftle_depends_on_the_start_time_when_unsteady():
    gyre = double_gyre()
    x, y = np.linspace(0.02, 1.98, 60), np.linspace(0.02, 0.98, 30)
    a = ftle(gyre, x, y, t0=0.0, T=15.0, dt=0.1)
    b = ftle(gyre, x, y, t0=2.5, T=15.0, dt=0.1)
    assert not np.allclose(np.nan_to_num(a.ftle), np.nan_to_num(b.ftle))


def test_backward_ftle_differs_from_forward():
    """Forward ridges repel, backward ridges attract -- they are different fields."""
    gyre = double_gyre()
    x, y = np.linspace(0.02, 1.98, 60), np.linspace(0.02, 0.98, 30)
    f = ftle(gyre, x, y, t0=5.0, T=10.0, dt=0.1)
    b = ftle(gyre, x, y, t0=5.0, T=-10.0, dt=0.1)
    assert f.forward and not b.forward
    assert not np.allclose(np.nan_to_num(f.ftle), np.nan_to_num(b.ftle))


def test_ridges_mask_selects_the_strongest(steady_ftle):
    r = steady_ftle.ridges(percentile=90)
    assert r.dtype == bool
    assert 0.05 < r.mean() < 0.2


def test_zero_T_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        ftle(double_gyre(), np.linspace(0, 1, 5), np.linspace(0, 1, 5), 0.0, 0.0, 0.1)


def test_2d_seed_axes_rejected():
    X = np.ones((4, 4))
    with pytest.raises(ValueError, match="1-D"):
        ftle(double_gyre(), X, X, 0.0, 1.0, 0.1)


def test_ftle_runs_on_a_real_dataset(scattered_case):
    """End to end through Dataset -> FlowField -> advect -> FTLE."""
    ds = Dataset.from_dataset(scattered_case)
    x, y = np.linspace(0.0, 4.0, 25), np.linspace(-1.0, 1.0, 15)
    res = ftle(ds, x, y, t0=0.5, T=1.5, dt=0.1, element_size=0.25)
    assert res.ftle.shape == (15, 25)
    assert np.isfinite(res.ftle).any()
