"""
Shared synthetic fixtures.

Real OpenFOAM data is far too slow and too large for a test suite, so everything
here is analytic: fields whose exact answer is known in closed form, on meshes
that are deliberately awkward (non-uniform cell volumes, scattered points) so
they exercise the same code paths a body-fitted CFD mesh does.
"""

import numpy as np
import pytest
import xarray as xr

# A freestream at 30 degrees, matching the AoA=30 case the library was built for.
AOA = 30.0
U_INF = (np.cos(np.radians(AOA)), np.sin(np.radians(AOA)))


def _dataset(x, y, u, v, p, times, V):
    """Assemble the (time, cell, comp) layout that read_foamcase produces."""
    n_t, n_c = len(times), len(x)
    U = np.zeros((n_t, n_c, 3))
    U[..., 0], U[..., 1] = u, v
    return xr.Dataset(
        {"p": (("time", "cell"), np.broadcast_to(p, (n_t, n_c)).copy()),
         "U": (("time", "cell", "comp"), U)},
        coords={"time": times, "x": ("cell", x), "y": ("cell", y), "V": ("cell", V)},
        attrs={"length_scale": 1.0},
    )


@pytest.fixture
def uniform_case():
    """
    Uniform flow at 30 degrees on a scattered mesh.

    Every operation has a closed-form answer: interpolation must return the
    freestream everywhere, a particle must travel exactly U*dt, and rotating by
    30 degrees must map the velocity onto (|U|, 0).
    """
    rng = np.random.default_rng(1)
    n_c = 1200
    x = rng.uniform(-2, 10, n_c)
    y = rng.uniform(-3, 3, n_c)
    V = 10.0 ** rng.uniform(-3, 0, n_c)          # 1000x spread, like a real mesh
    times = np.linspace(0.0, 2.0, 5)
    u = np.full(n_c, U_INF[0])
    v = np.full(n_c, U_INF[1])
    return _dataset(x, y, u, v, np.zeros(n_c), times, V)


@pytest.fixture
def scattered_case():
    """
    A travelling wave on a scattered, non-uniform mesh -- something with real
    spatial and temporal structure for crop/rotate/stepping tests.
    """
    rng = np.random.default_rng(2)
    n_c = 1500
    x = rng.uniform(-2, 10, n_c)
    y = rng.uniform(-3, 3, n_c)
    V = 10.0 ** rng.uniform(-3, 0, n_c)
    times = np.linspace(0.0, 4.0, 9)

    k, omega = 1.5, 2.0
    phase = k * x[None, :] - omega * times[:, None]
    u = U_INF[0] + 0.2 * np.sin(phase)
    v = U_INF[1] + 0.2 * np.cos(phase)
    p = np.zeros(n_c)
    return _dataset(x, y, u, v, p, times, V)


@pytest.fixture
def gridded_case():
    """
    A structured lattice, so interpolation onto a Cartesian grid is exact and
    the grid extent can be checked against the known bounds.
    """
    xs = np.linspace(0.0, 4.0, 41)
    ys = np.linspace(-1.0, 1.0, 21)
    X, Y = np.meshgrid(xs, ys)
    x, y = X.ravel(), Y.ravel()
    times = np.linspace(0.0, 1.0, 3)
    # linear in x and y -> linear interpolation must reproduce it exactly
    u = np.broadcast_to(1.0 + 0.5 * x, (len(times), x.size)).copy()
    v = np.broadcast_to(-0.25 * y, (len(times), x.size)).copy()
    V = np.full(x.size, 0.01)
    return _dataset(x, y, u, v, 2.0 * x, times, V)
