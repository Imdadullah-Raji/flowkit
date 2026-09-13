"""
Volume-weighted proper orthogonal decomposition (method of snapshots).

On an unstructured mesh the physically meaningful inner product is an integral
over the domain,

    <a, b> = sum_i a_i b_i V_i

not the plain dot product. Cell volumes in this mesh span nearly six orders of
magnitude, so an unweighted POD is dominated by the refined near-body region
simply because it holds more cells per unit area. Every inner product here is
volume weighted.
"""

import numpy as np
import xarray as xr
from scipy.linalg import eigh

from flowkit.dataprocessing import Dataset, Snapshot


class MissingCellVolumes(Exception):
    pass


class PODResult:
    """
    Spatial modes, temporal coefficients and energies from a volume-weighted POD.

    Modes are orthonormal in the volume-weighted inner product, i.e.
    Phi.T @ diag(w) @ Phi == I, with w = tile(V, 2) over the stacked (u, v) state.

    A snapshot is reconstructed as

        u(t) = mean_u + sum_k a_k(t) * modes[k, :, 0]
        v(t) = mean_v + sum_k a_k(t) * modes[k, :, 1]
    """

    def __init__(self, modes, energies, coefficients, mean_u, mean_v,
                 x, y, V, times):
        self.modes = modes                  # (n_modes, n_cells, 2)
        self.energies = energies            # (n_modes,)
        self.coefficients = coefficients    # (n_times, n_modes)
        self.mean_u = mean_u
        self.mean_v = mean_v
        self.x = x
        self.y = y
        self.V = V
        self.times = times

    @property
    def n_modes(self):
        return self.modes.shape[0]

    @property
    def n_cells(self):
        return self.modes.shape[1]

    def energy_fraction(self):
        """Fraction of total fluctuation energy carried by each mode."""
        return self.energies / self.energies.sum()

    def cumulative_energy(self):
        return np.cumsum(self.energy_fraction())

    def n_modes_for(self, fraction=0.99):
        """Number of modes needed to capture `fraction` of the energy."""
        return int(np.searchsorted(self.cumulative_energy(), fraction) + 1)

    def weighted_inner(self, a, b):
        """
        Volume-weighted inner product of two (n_cells, 2) fields.

        Use this rather than flattening by hand: `modes` is laid out
        (n_modes, n_cells, 2), so a flat `np.tile(V, 2)` silently applies the
        wrong weight to every entry. The weight belongs to the cell axis.
        """
        return float(np.einsum("cd,c,cd->", np.asarray(a), self.V, np.asarray(b)))

    def gram(self, k=None):
        """
        Weighted Gram matrix of the leading k modes. Should be the identity --
        a cheap check that a decomposition came out clean.
        """
        k = self.n_modes if k is None else min(k, self.n_modes)
        m = self.modes[:k]
        return np.einsum("kcd,c,jcd->kj", m, self.V, m)

    def mode_snapshot(self, k) -> Snapshot:
        """
        Mode k as a Snapshot, so it can be plotted or interpolated with the
        existing machinery (scatter, interpolate_on_grid, ...).

        The p field carries the mode's local energy density, which is what you
        usually want as the background of a mode plot.
        """
        u, v = self.modes[k, :, 0], self.modes[k, :, 1]
        return Snapshot(self.x, self.y, u, v, u**2 + v**2)

    def mean_snapshot(self) -> Snapshot:
        speed = np.hypot(self.mean_u, self.mean_v)
        return Snapshot(self.x, self.y, self.mean_u, self.mean_v, speed)

    def reconstruct(self, n_modes=None, time_index=None):
        """
        Rebuild (u, v) from the leading `n_modes`.

        With `time_index` given, returns arrays of shape (n_cells,) for that
        snapshot; otherwise (n_times, n_cells) for all of them.
        """
        k = self.n_modes if n_modes is None else min(n_modes, self.n_modes)
        a = self.coefficients[:, :k]
        if time_index is not None:
            a = a[time_index][None, :]
        u = self.mean_u + a @ self.modes[:k, :, 0]
        v = self.mean_v + a @ self.modes[:k, :, 1]
        return (u[0], v[0]) if time_index is not None else (u, v)

    def __repr__(self):
        f = self.energy_fraction()
        return (f"<PODResult {self.n_modes} modes, {self.n_cells} cells, "
                f"{len(self.times)} snapshots, "
                f"mode0={100*f[0]:.1f}% mode1={100*f[1]:.1f}%>")


def _state_matrix(ds: xr.Dataset):
    """Velocity as (n_times, n_cells, 2), loaded once."""
    U = ds["U"].values
    return np.ascontiguousarray(U[..., :2])


def pod(dataset, times=None, n_modes=None, subtract_mean=True,
        drop_initial=True, chunk=4096) -> PODResult:
    """
    Volume-weighted POD of the velocity fluctuations.

    Parameters
    ----------
    dataset : Dataset or xr.Dataset
        Must carry the 'V' cell coordinate -- see flowkit.io.attach_volumes.
    times : array-like or slice, optional
        Snapshots to use. Default is all of them.
    n_modes : int, optional
        Keep only the leading n_modes. Default keeps all.
    subtract_mean : bool
        Decompose fluctuations about the time mean (the usual choice).
    drop_initial : bool
        Drop a t=0 snapshot that is separated from the rest by a large gap. A
        uniform initial condition sitting 120 time units before the first real
        sample would otherwise distort the mean and add a spurious mode.
    chunk : int
        Cells per block when accumulating the correlation matrix.
    """
    ds = dataset.data if isinstance(dataset, Dataset) else dataset

    if "V" not in ds.coords:
        raise MissingCellVolumes(
            "Dataset has no 'V' cell coordinate. Either reconvert the case "
            "(read_foamcase now reads volumes) or, for an existing NetCDF:\n"
            "    from flowkit.io import attach_volumes\n"
            "    ds = attach_volumes(ds, casepath)"
        )

    if times is not None:
        ds = ds.isel(time=times) if isinstance(times, slice) else ds.sel(time=times)

    t = ds["time"].values
    if drop_initial and t.size > 2:
        gaps = np.diff(t)
        # first gap wildly out of scale with the rest -> leading outlier
        if gaps[0] > 10 * np.median(gaps[1:]):
            ds = ds.isel(time=slice(1, None))
            t = ds["time"].values

    V = np.asarray(ds["V"].values, dtype=float)
    Uv = _state_matrix(ds)                         # (n_t, n_c, 2)
    n_t, n_c, _ = Uv.shape
    if n_t < 2:
        raise ValueError(f"POD needs at least 2 snapshots, got {n_t}.")

    mean = Uv.mean(axis=0) if subtract_mean else np.zeros((n_c, 2))
    Uv -= mean                                     # in place: fluctuations

    # Correlation matrix C = X^T diag(w) X / n_t, accumulated over cell blocks
    # so the weighted copy of X never exists in full.
    C = np.zeros((n_t, n_t))
    for i in range(0, n_c, chunk):
        sl = slice(i, min(i + chunk, n_c))
        w = V[sl]
        for c in (0, 1):
            blk = Uv[:, sl, c]                     # (n_t, chunk)
            C += (blk * w) @ blk.T
    C /= n_t
    C = 0.5 * (C + C.T)                            # kill round-off asymmetry

    lam, Z = eigh(C)
    order = np.argsort(lam)[::-1]
    lam, Z = lam[order], Z[:, order]

    keep = n_t if n_modes is None else min(n_modes, n_t)
    # discard numerically null directions
    tol = max(lam[0], 0.0) * 1e-12
    keep = min(keep, int(np.sum(lam > tol)))
    lam, Z = lam[:keep], Z[:, :keep]

    # Phi_k = X Z_k / sqrt(n_t lam_k)  ->  Phi^T W Phi = I
    scale = 1.0 / np.sqrt(n_t * lam)
    modes = np.einsum("tcd,tk->kcd", Uv, Z) * scale[:, None, None]
    coeffs = Z * np.sqrt(n_t * lam)                # (n_t, keep)

    return PODResult(
        modes=modes,
        energies=lam,
        coefficients=coeffs,
        mean_u=mean[:, 0].copy(),
        mean_v=mean[:, 1].copy(),
        x=np.asarray(ds["x"].values),
        y=np.asarray(ds["y"].values),
        V=V,
        times=t,
    )
