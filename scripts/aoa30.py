"""Convert the AoA=30 case to NetCDF and eyeball one snapshot."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowkit.cases import case

CASE = case("re500_aoa30")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true",
                    help="read the OpenFOAM case and write the NetCDF cache")
    ap.add_argument("--time", type=float, default=None,
                    help="time to plot (default: the last one)")
    args = ap.parse_args()

    if args.convert:
        print("converting ->", CASE.convert(overwrite=True))

    ds = CASE.dataset()
    t = args.time if args.time is not None else float(ds.times[-1])
    snap = ds.snapshot(t)
    print(ds)
    print(snap)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.scatter(snap.x, snap.y, c=snap.u, s=1, cmap="RdBu_r")
    ax.set(xlabel="x/c", ylabel="y/c", aspect="equal", title=f"u at t={snap.time:g}")
    fig.colorbar(im, ax=ax, label="u")
    fig.tight_layout()
    out = CASE.figure(f"aoa30_u_t{snap.time:g}.png")
    fig.savefig(out, dpi=130)
    print("figure ->", out)
