# flowkit

A small research library for post-processing 2-D OpenFOAM simulations and doing
Lagrangian analysis on them — reading a case into `xarray`, slicing it into
snapshots, interpolating scattered cell data onto a Cartesian grid, and
advecting particles through the resulting velocity field (the groundwork for
FTLE / LCS computation).

> **Status: early research code.** The API is unstable, there are no tests, and
> several documented rough edges are listed in `ISSUES.md`. Read that file
> before relying on any result.

---

## Requirements

- Python ≥ 3.10
- [`fluidfoam`](https://fluidfoam.readthedocs.io) — reads OpenFOAM mesh/field files
- `numpy`, `scipy`, `xarray`, `netCDF4`
- `matplotlib` (only for the plotting scripts)

## Installation

```bash
git clone <this-repo> flowkit
cd flowkit
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core
pip install -e ".[viz]"     # + matplotlib for scripts/
```

The importable package is currently named `src` (see `ISSUES.md`, issue 1), so
all imports look like `from src.io import ...`. Everything also works without
installing, as long as you run from the repository root.

---

## Concepts

The library is three layers, each in its own module:

| Layer | Module | Type | Holds |
|---|---|---|---|
| I/O | `src.io` | functions | OpenFOAM case → `xarray.Dataset` → NetCDF |
| Fields | `src.dataprocessing` | `Dataset`, `Snapshot`, `CartesianSnapshot` | the whole time series, one instant, one instant on a regular grid |
| Particles | `src.lagrangian` | `Particles` | positions advected through a `CartesianSnapshot` |

```
OpenFOAM case ──read_foamcase──> xr.Dataset ──Dataset.from_*──> Dataset
                                     │                            │
                              foam_to_netcdf                  .snapshot(t)
                                     ↓                            ↓
                                  file.nc                     Snapshot  (scattered cell data)
                                                                  │
                                                    .interpolateOnCartesianGrid()
                                                                  ↓
                                                          CartesianSnapshot  (regular grid + interpolators)
                                                                  │
                                                            .sample(x, y)
                                                                  ↓
                                                            Particles.step()
```

Data is assumed **two-dimensional**: the mesh `z` coordinate is discarded and
only the `u`, `v` velocity components are used.

---

## Quickstart

### 1. Convert an OpenFOAM case once

Reading a case walks every time directory and parses every field, which is slow.
Do it once and cache to NetCDF:

```python
from src.io import foam_to_netcdf

foam_to_netcdf(
    casepath="/home/me/OpenFOAM/run/sq_new",
    outpath="data/square.nc",
    patch="square",          # boundary patch used to compute the body length scale
)
```

`patch` names the boundary of the immersed body (an airfoil, a cylinder, a
square). Its convex-hull diameter is stored on the dataset as the attribute
`length_scale`, which lets you later crop in body-lengths instead of metres.
The default is `'airfoil'` — pass your own patch name if that is not it.

### 2. Load and take a snapshot

```python
from src.dataprocessing import Dataset

ds = Dataset.from_netcdf("data/square.nc")
print(ds.times)              # array of available times
print(ds.length_scale)       # body length scale read back from the file attrs

snap = ds.snapshot(20.0)     # nearest-time Snapshot; pass a time that exists
```

A `Snapshot` exposes the scattered per-cell arrays `x, y, u, v, p` and the
bounding box `xmin, xmax, ymin, ymax`. Snapshots created through a `Dataset`
remember where they came from, so you can walk the time axis:

```python
nxt  = snap.next()
prev = snap.previous()
```

### 3. Crop to a region of interest

Crop in absolute coordinates, or in multiples of the body length scale:

```python
wake     = ds.crop(xmin=-0.5, xmax=4.0, ymin=-1.0, ymax=1.0)
wake_rel = ds.crop_relative(-2, 8, -2, 2)            # ×L about the origin
wake_rel = ds.crop_relative(-2, 8, -2, 2, origin=(0.1, 0.0), length_scale=0.5)
```

`Dataset.crop*` returns a new `Dataset` (nothing is copied — it is an `isel`
view). `Snapshot` crops in two steps, via an explicit boolean mask:

```python
m = snap.mask(xmin=-0.5, xmax=4.0, ymin=-1.0, ymax=1.0)   # all four are required
sub = snap.crop(m)
```

### 4. Interpolate onto a Cartesian grid

Particle advection needs a field that can be evaluated at arbitrary points, so
the scattered cell data is first resampled onto a regular grid:

```python
field = sub.interpolateOnCartesianGrid(element_size=0.01)

field.x, field.y            # 1-D grid axes
field.u, field.v, field.p   # 2-D arrays, shape (len(y), len(x))
field.elementSize           # grid spacing

u, v, p = field.sample(0.3, 0.15)          # scalars
u, v, p = field.sample(xq_array, yq_array) # broadcast, NaN outside the grid
```

Points outside the grid return `NaN` rather than raising. **Always pass
`element_size` explicitly** — the automatic default is a poor estimate and can
produce an enormous grid (`ISSUES.md`, issue 6).

### 5. Advect particles

```python
import numpy as np
from src.lagrangian import Particles

X, Y = np.meshgrid(np.arange(-0.5, 4.0, 0.02),
                   np.arange(-1.0, 1.0, 0.02))
p = Particles(X.ravel(), Y.ravel())

p = p.step(field, dt=0.01, method="Euler")   # returns a NEW Particles
print(p.n_particles, p.posx, p.posy)
```

`step` samples the velocity at each particle position, takes one explicit Euler
step, and drops any particle that left the field (its sample was `NaN`). Euler
is the only method implemented. `Particles` is immutable-by-convention: each
step returns a new object.

`Particles(X, Y, mask=...)` removes particles where `mask` is `True` — use it to
exclude points inside a solid body before starting integration.

### 6. Flow map and FTLE

FTLE is **not** in the library yet; it is assembled from the pieces above. The
shape of the calculation:

```python
p = Particles(X.ravel(), Y.ravel())
for t in times:                                    # integrate forward in time
    field = ds.snapshot(t).crop(m).interpolateOnCartesianGrid(element_size=h)
    p = p.step(field, dt=dt, method="Euler")

# flow map, reshaped back onto the seeding grid
fx = p.posx.reshape(X.shape)
fy = p.posy.reshape(X.shape)

# Jacobian by finite differences, then the largest singular value
dfxdy, dfxdx = np.gradient(fx, h)
dfydy, dfydx = np.gradient(fy, h)
...                                                # eigvalsh(J.T @ J), sqrt, /T
```

Note the pitfall: `Particles.step` silently discards escaped particles, so
`p.posx` will not generally have the same length as `X.ravel()` and the reshape
above will fail once anything leaves the domain. This is issue 12 in
`ISSUES.md`, and it is the main thing blocking a working FTLE routine.

---

## API reference

### `src.io`

| Function | Description |
|---|---|
| `read_foamcase(casepath, patch='airfoil')` | Parse every numeric time directory of an OpenFOAM case into an `xarray.Dataset` with variables `p` (`time`, `cell`) and `U` (`time`, `cell`, `comp`), coords `time`, `x`, `y`, and attrs `length_scale`, `source_case`. |
| `foam_to_netcdf(casepath, outpath, patch='airfoil')` | `read_foamcase` + `to_netcdf`. Returns `outpath`. |
| `body_length_scale(casepath, patch='airfoil')` | Maximum chord across the convex hull of the named boundary patch, in the mesh's units. |

### `src.dataprocessing.Dataset`

| Member | Description |
|---|---|
| `Dataset(data, length_scale=None)` | Wrap an `xarray.Dataset` directly. |
| `Dataset.from_foam(casepath, patch='airfoil')` | Read an OpenFOAM case. |
| `Dataset.from_netcdf(source)` | Open a NetCDF file written by `foam_to_netcdf`. |
| `Dataset.from_dataset(xr_ds)` | Wrap an existing `xarray.Dataset`, taking `length_scale` from its attrs. |
| `.data` | The underlying `xarray.Dataset`. |
| `.times` | Array of times. |
| `.length_scale` | Body length scale, or `None`. |
| `.snapshot(time)` | A `Snapshot` linked back to this dataset. |
| `.snapshots(times=None)` | Generator of snapshots (currently broken when `times` is omitted — issue 3). |
| `.crop(xmin, xmax, ymin, ymax)` | New `Dataset` restricted to a box; every bound is optional. |
| `.crop_relative(xmin, xmax, ymin, ymax, length_scale=None, origin=(0,0))` | Same, with bounds given in multiples of the length scale. |

### `src.dataprocessing.Snapshot`

Constructed as `Snapshot(x, y, p, u, v, dataset=None, index=None)` — note the
`p, u, v` ordering, which differs from the `u, v, p` that `sample()` returns.

| Member | Description |
|---|---|
| `Snapshot.from_dataset(time, xr_ds)` | Nearest-time slice, linked to the dataset. |
| `Snapshot.from_xarray(field)` | From an already-selected slice; not linked, so `next`/`previous` will raise. |
| `.x .y .u .v .p` | 1-D per-cell arrays. |
| `.xmin .xmax .ymin .ymax` | Bounding box. |
| `.next()` / `.previous()` | Adjacent snapshot in time; raises `NotAssociatedWithDataset` if unlinked. No bounds checking (issue 4). |
| `.mask(xmin, xmax, ymin, ymax)` | Boolean per-cell mask; all four bounds required. |
| `.crop(mask)` | New `Snapshot` from a boolean mask. Loses the dataset link. |
| `.constructCartesianGrid(element_size=None)` | The `(x, y)` axes that interpolation would use. |
| `.interpolateOnCartesianGrid(element_size=None)` | Linear `griddata` resampling → `CartesianSnapshot`. |

### `src.dataprocessing.CartesianSnapshot`

| Member | Description |
|---|---|
| `CartesianSnapshot(x, y, p, u, v)` | `x`, `y` are 1-D axes; the fields must be `(len(y), len(x))`. |
| `.x .y .p .u .v .elementSize` | Grid and fields. |
| `.sample(xq, yq)` | Bilinear sample → `(u, v, p)`, `NaN` outside the grid. |

Exceptions: `InvalidSnapshotFormat` (defined, never raised) and
`NotAssociatedWithDataset`.

### `src.lagrangian.Particles`

| Member | Description |
|---|---|
| `Particles(X, Y, mask=None)` | Flat position arrays; `mask=True` entries are dropped. |
| `.posx .posy .n_particles` | Current state. |
| `.step(field, dt=None, method="Euler")` | One explicit Euler step through a `CartesianSnapshot`; returns a new `Particles` with escaped particles removed. `dt` is required in practice. |

---

## Repository layout

```
src/                 the library
  io.py              OpenFOAM → xarray → NetCDF
  dataprocessing.py  Dataset / Snapshot / CartesianSnapshot
  lagrangian.py      Particles
scripts/             ad-hoc driver scripts (not tests, not importable API)
  test.py            plots a snapshot and its successor
data/                NetCDF caches — gitignored (*.nc)
figures/             saved output figures
```

`scripts/test.py` is a scratch script with absolute paths hardcoded to a
now-renamed directory; edit `casepath`/`outpath` before running it.

## Conventions

- **2-D only.** `z` and `w` are dropped at read time.
- **Field ordering.** Constructors take `p, u, v`; `sample()` returns `u, v, p`.
- **Grid ordering.** `CartesianSnapshot` fields are indexed `[y, x]`, matching
  `np.meshgrid` default (`xy`) output.
- **Immutability.** `crop`, `step`, `next`, `previous` and friends all return
  new objects; nothing mutates in place.
- **Units** are whatever the OpenFOAM case used; nothing is normalised.
