# flowkit

A small research library for post-processing 2-D OpenFOAM simulations and doing
Lagrangian analysis on them — reading a case into `xarray`, slicing it into
snapshots, interpolating scattered cell data onto a Cartesian grid, and
advecting particles through the resulting velocity field (the groundwork for
FTLE / LCS computation).

> **Status: early research code.** The API is unstable, there are no tests, and
> several rough edges is present. 

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


---

## Concepts

The library is three layers, each in its own module:

| Layer | Module | Type | Holds |
|---|---|---|---|
| I/O | `flowkit.io` | functions | OpenFOAM case → `xarray.Dataset` → NetCDF |
| Fields | `flowkit.dataprocessing` | `Dataset`, `Snapshot`, `CartesianSnapshot` | the whole time series, one instant, one instant on a regular grid |
| Particles | `flowkit.lagrangian` | `Particles`, `FlowField`, `advect` | tracers advected through a time-interpolated field |
| LCS | `flowkit.ftle` | `ftle()`, `FTLEResult` | finite-time Lyapunov exponent and its ridges |
| Cases | `flowkit.cases` | `case()` | where each simulation lives |
| Modal | `flowkit.pod` | `pod()`, `PODResult` | volume-weighted POD of the velocity fluctuations |

```
OpenFOAM case ──read_foamcase──> xr.Dataset ──Dataset.from_*──> Dataset
                                     │                            │
                              foam_to_netcdf                  .snapshot(t)
                                     ↓                            ↓
                                  file.nc                     Snapshot  (scattered cell data)
                                                                  │
                                                    .interpolate_on_grid()
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
from flowkit.io import foam_to_netcdf

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

**Cell volumes.** POD needs them (nothing else does), and they are read into the
`V` cell coordinate at conversion time. Write them once — the mesh is static, so
a single copy in `constant/` is correct:

```bash
cd /path/to/case
postProcess -func writeCellVolumes -latestTime
mv <latestTime>/Vc constant/V
```

`-latestTime` matters: without it OpenFOAM writes the field into *every* time
directory. `read_cell_volumes` accepts either the OpenFOAM `Vc` name or a
renamed `V`, and looks in `constant/` before the latest time directory. A case
without volumes still converts — it just warns, and `pod()` later tells you how
to add them with `attach_volumes` rather than making you reconvert.

### 2. Load and take a snapshot

```python
from flowkit.dataprocessing import Dataset

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
field = sub.interpolate_on_grid(element_size=0.01)

field.x, field.y            # 1-D grid axes
field.u, field.v, field.p   # 2-D arrays, shape (len(y), len(x))
field.element_size           # grid spacing

u, v, p = field.sample(0.3, 0.15)          # scalars
u, v, p = field.sample(xq_array, yq_array) # broadcast, NaN outside the grid
```

Points outside the grid return `NaN` rather than raising. The default `element_size` is `sqrt(area / n_points)`, a sane 2-D estimate; pass
your own when you want a specific resolution.

### 5. Advect particles

```python
import numpy as np
from flowkit import FlowField, Particles, advect

seeds, shape = Particles.on_grid(np.arange(-0.5, 4.0, 0.02),
                                 np.arange(-1.0, 1.0, 0.02))
final = advect(wake, seeds, t0=162.2, T=4.0, dt=0.02, element_size=0.02)

fx = final.posx.reshape(shape)        # always valid -- see below
print(final)                          # <Particles 10000 total, 7834 active>
```

**The particle array never changes length.** A particle that leaves the domain
becomes `NaN` and stays `NaN`; it is not deleted. That is what keeps
`posx.reshape(shape)` valid for the whole integration, and a flow map is exactly
that correspondence between a particle's index and its seeding position.

`advect` accepts a `Dataset` (gridded and interpolated in time on the fly), a
`FlowField`, or a single `CartesianSnapshot` for a frozen field. A negative `T`
integrates backwards. `method="rk4"` is the default; `"euler"` is available and
roughly 10⁶× less accurate over one orbit of a rotating field.

`FlowField` blends the two bracketing snapshots rather than freezing one, so a
`dt` smaller than the snapshot spacing actually resolves the time variation, and
it caches the unstructured→Cartesian interpolation per snapshot.

### 6. Work in a rotated frame

A wake that convects at an angle does not fit an axis-aligned box. Rotate the
domain into the streamwise frame first, then crop normally:

```python
aoa = np.degrees(np.arctan2(0.5, 0.866025404))    # from 0/U internalField -> 30.0
wake = ds.rotate(aoa).crop_relative(-1, 9, -2, 2)
```

`rotate()` rotates the **velocity components as well as the coordinates**, so
advection stays correct; rotating only `x`/`y` would leave every plot looking
right while integrating in the wrong direction. It records `frame_angle` and
`frame_origin` in the dataset attrs.

For the same 10x4 chord wake region on the AoA=30 case, this is not a cosmetic
choice:

| | bounding box | grid @ h=0.02 | NaN |
|---|---|---|---|
| lab frame (diagonal) | 10.48 x 8.36 | 0.22M pts | 54.7% |
| rotated frame | 10.00 x 4.00 | 0.10M pts | 1.0% |

Use `crop_rotated()` instead if you want the tilted selection but need to keep
lab-frame coordinates.

### 7. Proper orthogonal decomposition

POD needs an inner product, and on an unstructured mesh the correct one is an
integral, `<a,b> = sum a_i b_i V_i`. Cell volumes in a body-fitted mesh span
five orders of magnitude, so an unweighted POD is dominated by the refined
near-body region purely because it holds more cells per unit area.

Cell volumes therefore travel with the dataset as the `V` cell coordinate, which
means `crop`, `rotate` and friends carry them automatically. Write them once per
case (see **Cell volumes** below), then:

```python
from flowkit.pod import pod

res = pod(wake)                     # velocity fluctuations about the time mean
print(res)                          # <PODResult 412 modes, 36826 cells, ...>

res.energy_fraction()[:4]           # 0.227, 0.220, 0.177, 0.167
res.n_modes_for(0.99)               # 10
res.coefficients[:, 0]              # a_1(t)
u, v = res.reconstruct(n_modes=10)  # low-rank reconstruction
```

Modes are orthonormal in the volume-weighted inner product. Check it with
`res.gram()`, and use `res.weighted_inner(a, b)` rather than flattening by hand
-- `modes` is `(n_modes, n_cells, 2)`, so a flat `np.tile(V, 2)` applies the
wrong weight to every entry (`np.repeat(V, 2)` is the flat equivalent).

Any mode can be handed back to the rest of the library for plotting:

```python
snap = res.mode_snapshot(0)                      # a Snapshot
field = snap.interpolate_on_grid(0.02)    # ... or interpolate it
```

Results are saved as NetCDF, so a decomposition is computed once:

```python
res.save("data/pod_aoa30.nc")                 # 210 MB, all 412 modes
res.save("data/pod_aoa30_20.nc", n_modes=20)  #  11 MB, well past the 99% mark
res = PODResult.load("data/pod_aoa30.nc")
```

The file is self-describing and lazily readable — `xr.open_dataset(f)["energies"]`
returns the spectrum without touching the 210 MB of modes. It also records what
was decomposed: `source_case`, `frame_angle`, `subtract_mean`, `drop_initial`,
and `n_modes_computed`, so a truncated file knows it is truncated
(`res.truncated`). `dtype="float32"` halves the size, at the cost of taking
`gram()` from 1e-15 to ~1e-7 — fine for plotting, not for verification.

`scripts/pod_aoa30.py` runs the whole pipeline and self-verifies. On the AoA=30
wake (36826 cells, 1289 snapshots, ~31 s) the modes come out in near-degenerate
pairs -- 22.7/22.0%, 17.6/16.7%, 8.0/7.9% -- the signature of periodic vortex
shedding, with `a_1` and `a_2` sharing one frequency (St = 0.319) and 90 degrees
apart. Ten modes carry 99% of the energy.

### 8. Flow map and FTLE

```python
from flowkit import ftle

res = ftle(wake, x, y, t0=162.2, T=4.0, dt=0.02, element_size=0.02)

res.ftle            # (ny, nx) -- ln(sigma)/|T|
res.flowmap_x       # final positions, grid-shaped, NaN where escaped
res.ridges(90)      # boolean mask of the strongest 10% -- LCS candidates
res.escaped_fraction
```

A negative `T` gives the backward-time FTLE, whose ridges are attracting rather
than repelling structures.

Validated two ways. A uniform strain field `u = (ax, -ay)` has the closed-form
answer `FTLE = a` everywhere, and the implementation reproduces it to better
than **3e-12**.
The steady double gyre — available as `flowkit.double_gyre()` — puts its
separatrix at exactly `x = 1`, and the ridge lands within 0.05 of it; with
`epsilon=0.25` the ridge meanders off, as the canonical unsteady case should.

`scripts/ftle_aoa30.py` runs it on the wake.

## API reference

### `flowkit.io`

| Function | Description |
|---|---|
| `read_foamcase(casepath, patch='airfoil')` | Parse every numeric time directory of an OpenFOAM case into an `xarray.Dataset` with variables `p` (`time`, `cell`) and `U` (`time`, `cell`, `comp`), coords `time`, `x`, `y`, and attrs `length_scale`, `source_case`. |
| `foam_to_netcdf(casepath, outpath, patch='airfoil')` | `read_foamcase` + `to_netcdf`. Returns `outpath`. |
| `read_cell_volumes(casepath)` | Cell volumes from `constant/V` (or `Vc`), falling back to the latest time directory. Raises with the `postProcess` command if absent. |
| `attach_volumes(ds, casepath=None)` | Add the `V` coord to a dataset converted before volumes were read, without reconverting. Falls back to the `source_case` attr. |
| `body_length_scale(casepath, patch='airfoil')` | Maximum chord across the convex hull of the named boundary patch, in the mesh's units. |

### `flowkit.dataprocessing.Dataset`

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
| `.snapshots(times=None)` | Generator over all times by default. |
| `.crop(xmin, xmax, ymin, ymax)` | New `Dataset` restricted to a box; every bound is optional. |
| `.crop_relative(xmin, xmax, ymin, ymax, length_scale=None, origin=(0,0))` | Same, with bounds given in multiples of the length scale. |

### `flowkit.dataprocessing.Snapshot`

Constructed as `Snapshot(x, y, u, v, p, dataset=None, index=None)` — the same
`u, v, p` ordering that `sample()` returns.

| Member | Description |
|---|---|
| `Snapshot.from_dataset(time, xr_ds)` | Nearest-time slice, linked to the dataset. |
| `Snapshot.from_xarray(field)` | From an already-selected slice; not linked, so `next`/`previous` will raise. |
| `.x .y .u .v .p` | 1-D per-cell arrays. |
| `.xmin .xmax .ymin .ymax` | Bounding box. |
| `.next()` / `.previous()` | Adjacent snapshot in time. `IndexError` at either end, `NotAssociatedWithDataset` if detached. |
| `.mask(...)` / `.mask_rotated(...)` | Boolean per-cell mask; every bound optional. |
| `.crop(xmin=None, ..., mask=None)` | Same signature as `Dataset.crop`; keeps the dataset link, so `next()` returns the same region. |
| `.cartesian_grid(element_size=None)` | The `(x, y)` axes that interpolation would use. |
| `.interpolate_on_grid(element_size=None)` | Linear resampling → `CartesianSnapshot`. One cached Delaunay triangulation serves all three fields. |

### `flowkit.dataprocessing.CartesianSnapshot`

| Member | Description |
|---|---|
| `CartesianSnapshot(x, y, u, v, p)` | `x`, `y` are 1-D axes; the fields must be `(len(y), len(x))`. |
| `.x .y .p .u .v .element_size` | Grid and fields. |
| `.sample(xq, yq)` | Bilinear sample → `(u, v, p)`, `NaN` outside the grid. |

Exception: `NotAssociatedWithDataset`.

### `flowkit.pod`

| Member | Description |
|---|---|
| `pod(dataset, times=None, n_modes=None, subtract_mean=True, drop_initial=True)` | Volume-weighted POD by the method of snapshots. Requires the `V` coord. |
| `PODResult.modes` | `(n_modes, n_cells, 2)`, orthonormal in the `V` inner product. |
| `.energies` / `.energy_fraction()` / `.cumulative_energy()` / `.n_modes_for(f)` | Spectrum. |
| `.coefficients` | `(n_times, n_modes)` temporal coefficients. |
| `.gram(k)` / `.weighted_inner(a, b)` | Weighted inner products, with the cell-axis layout handled for you. |
| `.mode_snapshot(k)` / `.mean_snapshot()` | As a `Snapshot`, for plotting and interpolation. |
| `.reconstruct(n_modes, time_index=None)` | Low-rank reconstruction. |
| `.save(path, n_modes=None, dtype=None)` / `PODResult.load(path)` | NetCDF persistence. |
| `.meta` / `.truncated` | Provenance, and whether modes were dropped on save. |

`drop_initial` discards a leading `t=0` snapshot separated from the rest by a
large gap -- a uniform initial condition would otherwise distort the mean and
add a spurious mode.

### `flowkit.cases`

| Member | Description |
|---|---|
| `case(name)` | Look up a registered `Case`; `cases()` lists them. |
| `Case.dataset()` | The cached NetCDF as a `Dataset`, cell volumes attached. |
| `Case.convert()` | Read the OpenFOAM case and write the NetCDF. |
| `Case.aoa` / `.speed` | Freestream angle in degrees and magnitude. |
| `Case.figure(name)` | Output path, directory created on demand. |

Cases are declared in `cases.toml` at the repo root (or `$FLOWKIT_CASES`), not
in the package — registering a simulation is a config edit, not a code edit:

```toml
[defaults]
patch = "airfoil"
u_inf = [1.0, 0.0]

[cases.re500_aoa30]
path  = "/home/raji/Research/Thesis/static_airfoil/re500_aoa30"
u_inf = [0.866025404, 0.5]
notes = "Static airfoil, Re=500, AoA=30. Sheds at St~0.319."
```

`[defaults]` applies to every case unless overridden. An unknown key is an error
rather than being silently ignored, so a typo'd `u_infinity` fails loudly.

NetCDF resolution order: an explicit `nc=` on the entry, then `$FLOWKIT_DATA`,
then `<repo>/data/`.

### `flowkit.ftle`

| Member | Description |
|---|---|
| `ftle(field, x, y, t0, T, dt, ...)` | FTLE over the grid spanned by 1-D axes `x`, `y`. |
| `flow_map(...)` | Just the advected positions, grid-shaped. |
| `double_gyre(A, epsilon, omega)` | The canonical analytic validation flow. |
| `FTLEResult.ftle / .flowmap_x / .flowmap_y` | The field and the map it came from. |
| `.ridges(percentile)` / `.escaped_fraction` / `.forward` | Inspection. |

### `flowkit.lagrangian`

| Member | Description |
|---|---|
| `Particles(x, y, mask=None)` | Tracers. `mask=True` blanks a particle (NaN) rather than removing it. |
| `Particles.on_grid(x, y)` | Seed on a meshgrid; returns `(particles, shape)`. |
| `.posx .posy .active .n_particles .n_active` | State. Length never changes. |
| `.step(field, dt, t=None, method="rk4")` | One step; `"rk4"` or `"euler"`. |
| `advect(field, particles, t0, T, dt, ...)` | Integrate over `T`; negative goes backwards. |
| `FlowField(dataset, element_size)` | Velocity at any `(x, y, t)`, cached per snapshot. |


---

## Repository layout

```
flowkit/                 the library
  io.py              OpenFOAM → xarray → NetCDF
  dataprocessing.py  Dataset / Snapshot / CartesianSnapshot
  lagrangian.py      Particles, FlowField, advect
  ftle.py            FTLE / LCS
  cases.py           case registry (reads ../cases.toml)
  pod.py             volume-weighted POD
cases.toml           the case registry
tests/               pytest suite
scripts/             ad-hoc driver scripts (not tests, not importable API)
  test.py            plots a snapshot and its successor
  pod_aoa30.py       POD of the AoA=30 wake, with self-verification
  ftle_aoa30.py      FTLE of the same wake
data/                NetCDF caches — gitignored (*.nc)
figures/             saved output figures
```

`scripts/test.py` is a scratch script with absolute paths hardcoded to a
now-renamed directory; edit `casepath`/`outpath` before running it.

## Conventions

- **2-D only.** `z` and `w` are dropped at read time.
- **Field ordering.** `u, v, p` everywhere — constructors and `sample()` alike.
- **Grid ordering.** `CartesianSnapshot` fields are indexed `[y, x]`, matching
  `np.meshgrid` default (`xy`) output.
- **Immutability.** `crop`, `step`, `next`, `previous` and friends all return
  new objects; nothing mutates in place.
- **Units** are whatever the OpenFOAM case used; nothing is normalised.
