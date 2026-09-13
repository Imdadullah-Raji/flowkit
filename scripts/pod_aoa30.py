"""
Volume-weighted POD of the Re=500, AoA=30 airfoil wake.

The wake convects at the freestream angle, so the domain is rotated into the
streamwise frame before cropping -- an axis-aligned box around a 30-degree wake
would be mostly empty.
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dataprocessing import Dataset
from src.io import attach_volumes
from src.pod import pod

CASE = "/home/raji/Research/Thesis/static_airfoil/re500_aoa30"
NC = "/home/raji/Research/analyze_cfd/data/re500_aoa30.nc"
FIGDIR = "/home/raji/Research/flowkit/figures"

# freestream direction from 0/U internalField -> the wake axis
U_INF = (0.866025404, 0.500000000)


def load(nc=NC, case=CASE, box=(-1, 9, -2, 2), every=1):
    aoa = np.degrees(np.arctan2(U_INF[1], U_INF[0]))
    ds = Dataset.from_netcdf(nc)
    ds = Dataset(attach_volumes(ds.data, case), length_scale=ds.length_scale)
    wake = ds.rotate(aoa).crop_relative(*box)
    if every > 1:
        wake = Dataset(wake.data.isel(time=slice(None, None, every)),
                       length_scale=wake.length_scale)
    return aoa, wake


def verify(res, wake):
    """The checks from the plan, on real data."""
    print("\n--- verification " + "-" * 46)

    G = res.gram(min(8, res.n_modes))
    off = np.abs(G - np.eye(G.shape[0])).max()
    print(f"1. orthonormality   max|Phi^T W Phi - I| = {off:.2e}   {'OK' if off < 1e-8 else 'FAIL'}")

    # align with the snapshots the POD actually used (drop_initial may cut t=0)
    U = wake.data.sel(time=res.times)["U"].values[..., :2]
    Up = U - U.mean(axis=0)
    direct = np.einsum("tcd,c->", Up**2, res.V) / len(res.times)
    rel = abs(res.energies.sum() - direct) / direct
    print(f"2. energy closure   sum(lambda)={res.energies.sum():.6e}  direct={direct:.6e}"
          f"  rel={rel:.2e}   {'OK' if rel < 1e-9 else 'FAIL'}")

    errs = []
    for k in (1, 2, 5, 10, 25):
        if k > res.n_modes:
            break
        u, v = res.reconstruct(n_modes=k)
        errs.append(np.einsum("tc,c->", (u - U[..., 0])**2 + (v - U[..., 1])**2, res.V))
    mono = all(a > b for a, b in zip(errs, errs[1:]))
    print(f"3. reconstruction   errors {['%.3e' % e for e in errs]}  monotone: {mono}")

    area = res.V.sum() / 0.1
    print(f"4. volumes          sum(V)/0.1 = {area:.3f} chord^2  "
          f"(V spans {res.V.max()/res.V.min():.3g}x)")

    # shedding pair: near-equal energy, same frequency, ~90 deg out of phase
    dt = np.diff(res.times).mean()
    a1, a2 = res.coefficients[:, 0], res.coefficients[:, 1]
    f = np.fft.rfftfreq(len(a1), dt)
    f1 = f[np.argmax(np.abs(np.fft.rfft(a1 - a1.mean())))]
    f2 = f[np.argmax(np.abs(np.fft.rfft(a2 - a2.mean())))]
    ratio = res.energies[1] / res.energies[0]
    corr = np.corrcoef(a1, a2)[0, 1]
    st = f1 * 1.0 / np.hypot(*U_INF)          # Strouhal on chord
    print(f"5. shedding pair    E2/E1 = {ratio:.3f}   f1 = {f1:.4f}, f2 = {f2:.4f}"
          f"   St = {st:.4f}")
    print(f"                    corr(a1,a2) = {corr:+.3f} (near 0 => ~90 deg apart)")
    return st


def plot(res, aoa, st):
    frac = 100 * res.energy_fraction()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    n = min(25, res.n_modes)
    ax[0].bar(np.arange(1, n + 1), frac[:n], color="#4878a8")
    ax[0].set(xlabel="mode", ylabel="energy [%]", title="POD energy spectrum")
    ax[1].plot(np.arange(1, n + 1), np.cumsum(frac)[:n], "o-", color="#4878a8")
    ax[1].axhline(99, ls="--", c="gray", lw=.8)
    ax[1].set(xlabel="modes", ylabel="cumulative energy [%]",
              title=f"{res.n_modes_for(0.99)} modes reach 99%")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/pod_energy.png", dpi=130)

    nm = min(6, res.n_modes)
    fig, axes = plt.subplots(nm, 1, figsize=(9, 2.1 * nm), sharex=True)
    for k, a in enumerate(np.atleast_1d(axes)):
        s = res.mode_snapshot(k)
        lim = np.percentile(np.abs(s.u), 99.5)
        im = a.scatter(s.x, s.y, c=s.u, s=1.2, cmap="RdBu_r", vmin=-lim, vmax=lim)
        a.set(ylabel="y/c", aspect="equal")
        a.set_title(f"mode {k+1}  --  {frac[k]:.2f}% of energy", fontsize=9)
        fig.colorbar(im, ax=a, pad=.01, label="$\\phi_u$")
    np.atleast_1d(axes)[-1].set_xlabel("streamwise x/c  (frame rotated %.0f deg)" % aoa)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/pod_modes.png", dpi=130)

    fig, ax = plt.subplots(figsize=(9, 3))
    for k in range(min(4, res.n_modes)):
        ax.plot(res.times, res.coefficients[:, k], lw=.9, label=f"$a_{k+1}$")
    ax.set(xlabel="t", ylabel="coefficient", title=f"temporal coefficients (St = {st:.3f})")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/pod_coefficients.png", dpi=130)
    print(f"\nfigures -> {FIGDIR}/pod_{{energy,modes,coefficients}}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1, help="use every Nth snapshot")
    ap.add_argument("--modes", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    aoa, wake = load(every=args.every)
    print(f"loaded: {wake.data.sizes['cell']} cells, {wake.data.sizes['time']} snapshots "
          f"(rotated {aoa:.1f} deg)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    res = pod(wake, n_modes=args.modes)
    print(f"pod: {res}  [{time.time()-t0:.1f}s]")

    st = verify(res, wake)
    plot(res, aoa, st)
