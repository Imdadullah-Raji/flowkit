"""
FTLE of the Re=500, AoA=30 airfoil wake.

Rebuilt as a driver over `flowkit.ftle`; the original `ftle_calc.py` /
`ftle_real.py` were never committed and survive only as bytecode.
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flowkit import ftle
from flowkit.cases import case

CASE = case("re500_aoa30")


def main(T=4.0, dt=0.02, h=0.02, box=(-1, 9, -2, 2), t0=None, element_size=0.02):
    wake = CASE.dataset().rotate(CASE.aoa).crop_relative(*box)
    times = wake.times
    t0 = float(times[len(times) // 3]) if t0 is None else t0
    print(f"{wake}\n  t0={t0:g}  T={T:g}  dt={dt:g}  seed h={h:g}")

    x = np.arange(box[0], box[1], h) * (CASE.dataset().length_scale or 1.0)
    y = np.arange(box[2], box[3], h) * (CASE.dataset().length_scale or 1.0)

    t = time.time()
    res = ftle(wake, x, y, t0=t0, T=T, dt=dt, element_size=element_size)
    print(f"  {res}  [{time.time()-t:.1f}s]")

    fig, ax = plt.subplots(figsize=(11, 4.4))
    lo, hi = np.nanpercentile(res.ftle, [2, 99.5])
    im = ax.pcolormesh(res.x, res.y, res.ftle, cmap="inferno", vmin=lo, vmax=hi,
                       shading="auto")
    ax.set(xlabel=f"streamwise x/c (frame rotated {CASE.aoa:.0f} deg)", ylabel="y/c",
           aspect="equal",
           title=f"{'Forward' if res.forward else 'Backward'} FTLE, "
                 f"t0={t0:.2f}, T={T:g}")
    fig.colorbar(im, ax=ax, label=r"$\sigma$  [1/t]", pad=0.01)
    fig.tight_layout()
    out = CASE.figure(f"ftle_T{T:g}_t{t0:.2f}.png")
    fig.savefig(out, dpi=140)
    print("  figure ->", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-T", type=float, default=4.0, help="integration time (negative = backward)")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--h", type=float, default=0.02, help="seed grid spacing, chords")
    ap.add_argument("--t0", type=float, default=None)
    args = ap.parse_args()
    main(T=args.T, dt=args.dt, h=args.h, t0=args.t0)
