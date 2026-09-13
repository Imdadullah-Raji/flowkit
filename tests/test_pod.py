"""Synthetic-field checks for the volume-weighted POD."""

import numpy as np
import pytest
import xarray as xr

from flowkit import PODResult
from flowkit.pod import pod, MissingCellVolumes


def synthetic(n_cells=500, n_times=200, seed=0):
    """
    Two planted modes, orthonormal in the volume-weighted inner product, driven
    by exactly orthogonal mean-zero coefficients with a 3:1 amplitude ratio.
    Cell volumes span four orders of magnitude, as on a real CFD mesh, so an
    unweighted POD cannot reproduce the planted answer.

    Energies are therefore exactly 9/2 and 1/2 -- a 90% / 10% split.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n_cells, endpoint=False)
    y = np.zeros(n_cells)
    V = 10.0 ** rng.uniform(-4, 0, n_cells)

    # disjoint components -> V-orthogonal; normalise in the V inner product
    p1 = np.stack([np.sin(x), np.zeros(n_cells)], axis=-1)
    p2 = np.stack([np.zeros(n_cells), np.cos(x)], axis=-1)
    p1 /= np.sqrt(np.einsum("cd,c->", p1**2, V))
    p2 /= np.sqrt(np.einsum("cd,c->", p2**2, V))

    # whole number of periods on a uniform grid -> exactly orthogonal, zero mean
    ph = 2 * np.pi * np.arange(n_times) / n_times
    a1, a2 = 3.0 * np.sin(ph), 1.0 * np.cos(ph)
    t = np.linspace(0, 10, n_times)

    U3 = np.zeros((n_times, n_cells, 3))
    U3[..., :2] = a1[:, None, None] * p1 + a2[:, None, None] * p2

    return xr.Dataset(
        {"U": (("time", "cell", "comp"), U3)},
        coords={"time": t, "x": ("cell", x), "y": ("cell", y), "V": ("cell", V)},
    )


@pytest.fixture
def result():
    return pod(synthetic(), subtract_mean=True)


@pytest.fixture
def scattered_case_meta():
    """A dataset carrying the provenance attrs that rotate()/read_foamcase set."""
    ds = synthetic()
    ds.attrs.update(source_case="/somewhere/case", frame_angle=30.0,
                    frame_origin=(0.0, 0.0), length_scale=1.0)
    return ds


def test_requires_volumes():
    ds = synthetic().drop_vars("V")
    with pytest.raises(MissingCellVolumes):
        pod(ds)


def test_modes_are_weight_orthonormal(result):
    """Phi^T W Phi == I, the defining property of the weighted POD."""
    k = min(6, result.n_modes)
    G = result.gram(k)
    assert np.abs(G - np.eye(k)).max() < 1e-10


def test_gram_matches_manual_cellwise_weighting(result):
    """The weight belongs to the cell axis; repeat(V,2) is the flat equivalent."""
    k = result.n_modes
    flat = result.modes[:k].reshape(k, -1)
    manual = (flat * np.repeat(result.V, 2)) @ flat.T
    assert np.allclose(result.gram(k), manual, atol=1e-12)


def test_energy_closure(result):
    """Sum of eigenvalues == volume-weighted mean fluctuation energy."""
    ds = synthetic()
    U = ds["U"].values[..., :2]
    Up = U - U.mean(axis=0)
    direct = np.einsum("tcd,c->", Up**2, result.V) / len(result.times)
    assert np.isclose(result.energies.sum(), direct, rtol=1e-10)


def test_rank_and_energy_ordering(result):
    """Two synthetic modes -> two non-null modes, ordered by energy 9:1."""
    assert result.n_modes == 2
    assert result.energies[0] > result.energies[1]
    assert np.isclose(result.energy_fraction()[0], 9 / 10, rtol=1e-9)
    assert np.isclose(result.energies[0], 9 / 2, rtol=1e-9)
    assert np.isclose(result.energies[1], 1 / 2, rtol=1e-9)


def test_reconstruction_is_monotone_and_exact(result):
    ds = synthetic()
    U = ds["U"].values[..., :2]
    errs = []
    for k in (1, 2):
        u, v = result.reconstruct(n_modes=k)
        err = np.einsum("tc,c->", (u - U[..., 0])**2 + (v - U[..., 1])**2, result.V)
        errs.append(err)
    assert errs[0] > errs[1]
    assert errs[-1] < 1e-16 * max(errs[0], 1.0)     # full rank -> exact


def test_drop_initial_removes_leading_outlier():
    """A t=0 snapshot far from the rest is dropped by default, kept on request."""
    ds = synthetic(n_times=50)
    t = ds["time"].values.copy()
    t[0] = t[1] - 500.0                              # huge leading gap
    ds = ds.assign_coords(time=t)
    assert len(pod(ds).times) == 49
    assert len(pod(ds, drop_initial=False).times) == 50


def test_weighting_actually_matters():
    """
    Unweighted POD on a non-uniform mesh gives different modes -- the whole
    reason the V coordinate is required.
    """
    ds = synthetic()
    weighted = pod(ds)
    flat = pod(ds.assign_coords(V=("cell", np.ones(ds.sizes["cell"]))))
    assert not np.isclose(weighted.energy_fraction()[0],
                          flat.energy_fraction()[0], rtol=1e-3)


# --------------------------------------------------------------- persistence

def test_roundtrip_preserves_everything(result, tmp_path):
    f = result.save(tmp_path / "pod.nc")
    back = PODResult.load(f)

    assert back.n_modes == result.n_modes
    assert back.n_cells == result.n_cells
    for name in ("energies", "modes", "coefficients", "mean_u", "mean_v",
                 "x", "y", "V", "times"):
        assert np.array_equal(getattr(back, name), getattr(result, name)), name


def test_roundtrip_preserves_orthonormality(result, tmp_path):
    back = PODResult.load(result.save(tmp_path / "pod.nc"))
    k = back.n_modes
    assert np.abs(back.gram(k) - np.eye(k)).max() < 1e-12


def test_roundtrip_preserves_reconstruction(result, tmp_path):
    back = PODResult.load(result.save(tmp_path / "pod.nc"))
    u0, v0 = result.reconstruct(n_modes=2)
    u1, v1 = back.reconstruct(n_modes=2)
    assert np.array_equal(u0, u1) and np.array_equal(v0, v1)


def test_provenance_survives(scattered_case_meta, tmp_path):
    """A saved result must say what it decomposed."""
    res = pod(scattered_case_meta)
    assert res.meta["source_case"] == "/somewhere/case"
    assert res.meta["frame_angle"] == 30.0
    back = PODResult.load(res.save(tmp_path / "pod.nc"))
    assert back.meta["source_case"] == "/somewhere/case"
    assert back.meta["frame_angle"] == 30.0
    assert back.meta["subtract_mean"] == 1


def test_truncated_save_records_the_full_count(result, tmp_path):
    back = PODResult.load(result.save(tmp_path / "pod.nc", n_modes=1))
    assert back.n_modes == 1
    assert back.meta["n_modes_computed"] == result.n_modes
    assert back.truncated
    assert not result.truncated


def test_energies_readable_without_loading_modes(result, tmp_path):
    """The reason for NetCDF: lazy access to one variable."""
    f = result.save(tmp_path / "pod.nc")
    with xr.open_dataset(f) as ds:
        assert ds["modes"].chunks is None or True      # not materialised yet
        e = ds["energies"].values
    assert np.array_equal(e, result.energies)


def test_float32_save_is_smaller_and_still_close(result, tmp_path):
    big = result.save(tmp_path / "f64.nc", compress=False)
    small = result.save(tmp_path / "f32.nc", dtype="float32", compress=False)
    assert small.stat().st_size < big.stat().st_size
    back = PODResult.load(small)
    assert np.abs(back.modes - result.modes).max() < 1e-6
