"""Particle advection: conservation, accuracy, and escape handling."""

import numpy as np
import pytest

from flowkit import Dataset, FlowField, Particles, advect
from tests.conftest import U_INF


class Rotation:
    """Solid-body rotation: exact circular orbits, so integrator error is measurable."""
    time_dependent = True

    def __init__(self, omega=1.0):
        self.omega = omega

    def sample(self, x, y, t):
        return -self.omega * np.asarray(y), self.omega * np.asarray(x)


class Bounded:
    """Uniform rightward flow that ends at x = 1 -- particles past it escape."""
    time_dependent = True

    def sample(self, x, y, t):
        x = np.asarray(x, dtype=float)
        inside = x <= 1.0
        return np.where(inside, 1.0, np.nan), np.where(inside, 0.0, np.nan)


# ------------------------------------------------------------- the core invariant

def test_particle_count_never_changes(uniform_case):
    """Issue #12: step() used to delete escaped particles and shorten the array."""
    p = Particles(np.linspace(0.0, 2.0, 50), np.zeros(50))
    for _ in range(10):
        p = p.step(Bounded(), dt=0.1, t=0.0)
        assert p.n_particles == 50


def test_escaped_particles_become_nan_and_stay_nan():
    p = Particles([0.5, 5.0], [0.0, 0.0])
    p = p.step(Bounded(), dt=0.1, t=0.0)
    assert not np.isnan(p.posx[0])
    assert np.isnan(p.posx[1])
    p = p.step(Bounded(), dt=0.1, t=0.0)
    assert np.isnan(p.posx[1])          # never resurrects
    assert p.n_active == 1


def test_grid_seeding_stays_reshapeable():
    """The property the flow map depends on."""
    x, y = np.linspace(0, 2, 7), np.linspace(-1, 1, 5)
    p, shape = Particles.on_grid(x, y)
    assert shape == (5, 7)
    p = advect(Bounded(), p, t0=0.0, T=1.0, dt=0.1)
    assert p.posx.reshape(shape).shape == shape


def test_excluded_particles_are_blanked_not_removed():
    mask = np.array([False, True, False])
    p = Particles([0.0, 0.5, 1.0], [0.0, 0.0, 0.0], mask=mask)
    assert p.n_particles == 3
    assert np.isnan(p.posx[1])
    assert p.n_active == 2


# --------------------------------------------------------------------- accuracy

def test_euler_is_exact_in_uniform_flow():
    p = Particles([0.0], [0.0])
    p = p.step(Rotation(omega=0.0), dt=0.5, t=0.0, method="euler")
    assert np.allclose([p.posx[0], p.posy[0]], [0.0, 0.0])


def test_uniform_flow_moves_by_u_times_dt(uniform_case):
    field = FlowField(Dataset.from_dataset(uniform_case), element_size=0.2)
    p = Particles([0.0], [0.0])
    p = p.step(field, dt=0.5, t=0.5, method="rk4")
    assert np.isclose(p.posx[0], 0.5 * U_INF[0], atol=1e-6)
    assert np.isclose(p.posy[0], 0.5 * U_INF[1], atol=1e-6)


def test_rk4_beats_euler_on_a_rotating_field():
    """One full orbit of solid-body rotation; the exact answer is the start point."""
    field, T = Rotation(omega=1.0), 2 * np.pi
    errs = {}
    for method in ("euler", "rk4"):
        p = advect(field, Particles([1.0], [0.0]), t0=0.0, T=T, dt=0.01, method=method)
        errs[method] = np.hypot(p.posx[0] - 1.0, p.posy[0])
    assert errs["rk4"] < errs["euler"] / 1000
    assert errs["rk4"] < 1e-8


def test_advect_lands_exactly_on_the_target_time():
    """dt need not divide T; the last step is trimmed."""
    field = Rotation(omega=0.0)
    p = advect(field, Particles([0.0], [0.0]), t0=0.0, T=1.0, dt=0.3)
    assert p.n_particles == 1


def test_backward_advection_undoes_forward():
    field = Rotation(omega=1.0)
    p0 = Particles([1.0, 0.3], [0.0, 0.7])
    fwd = advect(field, p0, t0=0.0, T=2.0, dt=0.005)
    back = advect(field, fwd, t0=2.0, T=-2.0, dt=0.005)
    assert np.allclose(back.posx, p0.posx, atol=1e-7)
    assert np.allclose(back.posy, p0.posy, atol=1e-7)


# ----------------------------------------------------------------- error paths

def test_unknown_method_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown method"):
        Particles([0.0], [0.0]).step(Rotation(), dt=0.1, method="verlet")


def test_missing_dt_raises():
    with pytest.raises(TypeError, match="dt is required"):
        Particles([0.0], [0.0]).step(Rotation(), dt=None)


def test_negative_dt_rejected():
    with pytest.raises(ValueError, match="dt must be positive"):
        advect(Rotation(), Particles([0.0], [0.0]), t0=0.0, T=1.0, dt=-0.1)


def test_mismatched_seed_shapes_rejected():
    with pytest.raises(ValueError, match="must match"):
        Particles([0.0, 1.0], [0.0])


# ------------------------------------------------------------------- FlowField

def test_flowfield_interpolates_between_snapshots(scattered_case):
    """A frozen snapshot would give the same answer at every t -- issue #14."""
    field = FlowField(Dataset.from_dataset(scattered_case), element_size=0.25)
    q = (np.array([2.0]), np.array([0.0]))
    t_a = float(scattered_case["time"].values[1])
    t_b = float(scattered_case["time"].values[2])
    u_a, _ = field.sample(*q, t_a)
    u_mid, _ = field.sample(*q, 0.5 * (t_a + t_b))
    u_b, _ = field.sample(*q, t_b)
    assert not np.isclose(u_a, u_b, atol=1e-6)
    assert min(u_a, u_b) - 1e-9 <= u_mid <= max(u_a, u_b) + 1e-9


def test_flowfield_grids_each_snapshot_once(scattered_case):
    field = FlowField(Dataset.from_dataset(scattered_case), element_size=0.25)
    a = field.frame(0)
    assert field.frame(0) is a          # cached, not re-interpolated
