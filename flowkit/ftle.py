"""
Finite-time Lyapunov exponent.

FTLE measures how fast neighbouring particles separate over a finite window, and
its ridges mark the Lagrangian coherent structures that organise a wake. It is
computed from the flow map: seed a grid, advect it, and look at how the grid
deforms.

    sigma  = largest singular value of grad(phi)
    FTLE   = ln(sigma) / |T|

This lives in the library rather than in a script on purpose -- the earlier
version of this calculation existed only as `scripts/ftle_calc.py` and
`scripts/ftle_real.py`, was never committed, and survives only as bytecode.
"""

import numpy as np

from flowkit.lagrangian import FlowField, Particles, advect


class FTLEResult:
    """The FTLE field on the seeding grid, plus the flow map it came from."""

    def __init__(self, x, y, ftle, flowmap_x, flowmap_y, t0, T):
        self.x = x                      # 1-D seeding axes
        self.y = y
        self.ftle = ftle                # (ny, nx)
        self.flowmap_x = flowmap_x      # (ny, nx) final positions
        self.flowmap_y = flowmap_y
        self.t0 = t0
        self.T = T

    @property
    def forward(self):
        return self.T > 0

    @property
    def escaped_fraction(self):
        return float(np.isnan(self.flowmap_x).mean())

    def ridges(self, percentile=90):
        """Boolean mask of the strongest FTLE values -- the LCS candidates."""
        finite = self.ftle[np.isfinite(self.ftle)]
        if finite.size == 0:
            return np.zeros_like(self.ftle, dtype=bool)
        return self.ftle >= np.percentile(finite, percentile)

    def __repr__(self):
        f = self.ftle[np.isfinite(self.ftle)]
        rng = f"{f.min():.3g}..{f.max():.3g}" if f.size else "all-nan"
        return (f"<FTLEResult {self.ftle.shape[1]}x{self.ftle.shape[0]} grid, "
                f"{'forward' if self.forward else 'backward'} T={self.T:g}, "
                f"range {rng}, {100*self.escaped_fraction:.1f}% escaped>")


def flow_map(field, x, y, t0, T, dt, method="rk4", element_size=None, mask=None):
    """
    Advect a grid of seeds and return their final positions, grid-shaped.

    Escaped particles come back as NaN rather than being dropped, so the result
    always has the shape of the seeding grid.
    """
    seeds, shape = Particles.on_grid(x, y, mask=mask)
    final = advect(field, seeds, t0, T, dt, method=method, element_size=element_size)
    return final.posx.reshape(shape), final.posy.reshape(shape)


def ftle(field, x, y, t0, T, dt, method="rk4", element_size=None, mask=None):
    """
    FTLE field over the grid spanned by the 1-D axes `x` and `y`.

    A negative `T` gives the backward-time FTLE, whose ridges are attracting
    structures; positive `T` gives repelling ones.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D seeding axes; use np.linspace")
    if T == 0:
        raise ValueError("T must be non-zero")

    fx, fy = flow_map(field, x, y, t0, T, dt, method=method,
                      element_size=element_size, mask=mask)
    return FTLEResult(x, y, _ftle_from_flowmap(fx, fy, x, y, T), fx, fy, t0, T)


def _ftle_from_flowmap(fx, fy, x, y, T):
    """
    FTLE from a flow map by finite differences.

    The Cauchy-Green tensor C = J^T J is 2x2 and symmetric, so its largest
    eigenvalue has a closed form -- no need to loop over the grid.
    """
    hx = float(x[1] - x[0]) if x.size > 1 else 1.0
    hy = float(y[1] - y[0]) if y.size > 1 else 1.0

    # np.gradient on a (ny, nx) array differentiates along y first, then x
    dfx_dy, dfx_dx = np.gradient(fx, hy, hx)
    dfy_dy, dfy_dx = np.gradient(fy, hy, hx)

    c11 = dfx_dx**2 + dfy_dx**2
    c12 = dfx_dx * dfx_dy + dfy_dx * dfy_dy
    c22 = dfx_dy**2 + dfy_dy**2

    tr, det = c11 + c22, c11 * c22 - c12**2
    disc = np.maximum(tr**2 / 4.0 - det, 0.0)
    lam_max = tr / 2.0 + np.sqrt(disc)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = 0.5 * np.log(lam_max) / abs(T)
    return np.where(np.isfinite(out), out, np.nan)


def double_gyre(A=0.1, epsilon=0.25, omega=2 * np.pi / 10.0):
    """
    The canonical analytic test flow for FTLE, on [0,2] x [0,1].

    With epsilon=0 it is steady and the gyre boundary sits exactly at x=1, which
    makes it a sharp check: the FTLE ridge must land there.
    """

    class DoubleGyre:
        time_dependent = True

        def sample(self, xq, yq, t):
            xq = np.asarray(xq, dtype=float)
            yq = np.asarray(yq, dtype=float)
            a = epsilon * np.sin(omega * t)
            f = a * xq**2 + (1.0 - 2.0 * a) * xq
            dfdx = 2.0 * a * xq + (1.0 - 2.0 * a)
            u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * yq)
            v = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * yq) * dfdx
            outside = (xq < 0) | (xq > 2) | (yq < 0) | (yq > 1)
            return np.where(outside, np.nan, u), np.where(outside, np.nan, v)

        def __repr__(self):
            return f"<DoubleGyre A={A} eps={epsilon} omega={omega:.4g}>"

    return DoubleGyre()
