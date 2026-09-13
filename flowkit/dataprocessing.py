import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import Delaunay

from flowkit.io import body_length_scale, read_foamcase


class NotAssociatedWithDataset(Exception):
    pass


class Dataset:
    """A time series of 2-D fields over an unstructured mesh."""

    def __init__(self, data: xr.Dataset, length_scale: float = None):
        self._data = data
        self.length_scale = length_scale

    @property
    def data(self):
        return self._data

    @property
    def times(self):
        return self._data['time'].values

    def snapshot(self, time):
        return Snapshot.from_dataset(time, self._data)

    def snapshots(self, times=None):
        times = self.times if times is None else times
        for t in times:
            yield self.snapshot(t)

    @classmethod
    def from_foam(cls, casepath, patch='airfoil'):
        data = read_foamcase(casepath, patch=patch)
        return cls(data, length_scale=data.attrs.get('length_scale'))

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset):
        return cls(dataset, dataset.attrs.get('length_scale'))

    @classmethod
    def from_netcdf(cls, source):
        ds = xr.open_dataset(source)
        return cls(ds, ds.attrs.get('length_scale'))

    def close(self):
        self._data.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _with(self, data):
        return type(self)(data, length_scale=self.length_scale)

    def crop(self, xmin=None, xmax=None, ymin=None, ymax=None) -> "Dataset":
        """Restrict to an axis-aligned box. Every bound is optional."""
        mask = _bounds_mask(self._data['x'].values, self._data['y'].values,
                            xmin, xmax, ymin, ymax)
        return self._with(self._data.isel(cell=mask))

    def crop_relative(self, xmin, xmax, ymin, ymax,
                      length_scale=None, origin=(0, 0)) -> "Dataset":
        """As crop, with bounds in multiples of the body length scale."""
        x0, y0 = origin
        L = self._length_scale_or_raise(length_scale)
        return self.crop(xmin=x0 + xmin * L, xmax=x0 + xmax * L,
                         ymin=y0 + ymin * L, ymax=y0 + ymax * L)

    def crop_rotated(self, xmin=None, xmax=None, ymin=None, ymax=None,
                     angle=0.0, origin=(0, 0), degrees=True) -> "Dataset":
        """
        Crop to a rectangle whose axes are rotated by `angle` about `origin`.

        Bounds are given in the rotated frame: xmin/xmax run along `angle`
        (streamwise), ymin/ymax across it. Coordinates stay in the lab frame --
        only the cell selection changes. Use .rotate() as well to express the
        region itself in the rotated frame.
        """
        mask = _rotated_bounds_mask(self._data['x'].values, self._data['y'].values,
                                    xmin, xmax, ymin, ymax,
                                    angle=angle, origin=origin, degrees=degrees)
        return self._with(self._data.isel(cell=mask))

    def crop_rotated_relative(self, xmin, xmax, ymin, ymax, angle=0.0,
                              length_scale=None, origin=(0, 0),
                              degrees=True) -> "Dataset":
        """As crop_rotated, with bounds in multiples of the body length scale."""
        L = self._length_scale_or_raise(length_scale)
        return self.crop_rotated(xmin=xmin * L, xmax=xmax * L,
                                 ymin=ymin * L, ymax=ymax * L,
                                 angle=angle, origin=origin, degrees=degrees)

    def rotate(self, angle, origin=(0, 0), degrees=True) -> "Dataset":
        """
        Re-express the dataset in a frame rotated by `angle` about `origin`.

        Rotates the x/y coordinates *and* the U components, so the data stays
        physically consistent and advection remains correct. Afterwards a region
        aligned with `angle` is axis-aligned, so plain crop() and
        interpolate_on_grid() work on it without waste.
        """
        ds = self._data.copy()
        xr_, yr_ = _rotate(ds['x'].values, ds['y'].values, angle, origin, degrees)

        U = np.array(ds['U'].values)
        U[..., 0], U[..., 1] = _rotate_vector(U[..., 0], U[..., 1], angle, degrees)

        ds = ds.assign_coords(x=('cell', xr_), y=('cell', yr_))
        ds['U'] = (self._data['U'].dims, U)
        ds.attrs['frame_angle'] = angle if degrees else np.degrees(angle)
        ds.attrs['frame_origin'] = tuple(origin)
        return self._with(ds)

    def _length_scale_or_raise(self, length_scale):
        L = length_scale if length_scale is not None else self.length_scale
        if L is None:
            raise ValueError(
                "No length_scale given and none stored on this Dataset "
                "(construct via from_foam, or pass length_scale= explicitly)."
            )
        return L

    def __len__(self):
        return self._data.sizes['time']

    def __repr__(self):
        t = self.times
        span = f"{t[0]:g}..{t[-1]:g}" if t.size else "empty"
        L = f"{self.length_scale:g}" if self.length_scale else "none"
        return (f"<Dataset {self._data.sizes['cell']} cells, "
                f"{t.size} times [{span}], L={L}>")


class Snapshot:
    """
    One instant of the flow on the unstructured mesh.

    Constructed as Snapshot(x, y, u, v, p) -- the same u, v, p order that
    CartesianSnapshot.sample() returns.

    A snapshot made through a Dataset remembers where it came from, including
    any crop, so next()/previous() return the same region at adjacent times.
    """

    def __init__(self, x, y, u, v, p, dataset=None, index=None, cells=None):
        self._x = np.asarray(x)
        self._y = np.asarray(y)
        self._u = np.asarray(u)
        self._v = np.asarray(v)
        self._p = np.asarray(p)
        self._dataset = dataset
        self._index = index
        self._cells = cells          # indices into the parent dataset, or None
        self._tri = None             # Delaunay, built once on demand

    # ---------------------------------------------------------- constructors

    @classmethod
    def from_xarray(cls, f: xr.DataArray):
        """From an already-selected time slice. Not linked to a dataset."""
        return cls(f['x'].values, f['y'].values,
                   f['U'].values[:, 0], f['U'].values[:, 1], f['p'].values)

    @classmethod
    def from_dataset(cls, time, dataset: xr.Dataset, cells=None):
        """
        Nearest-time slice, linked back to `dataset`.

        The index and the slice come from a single nearest-match lookup, so they
        cannot disagree.
        """
        idx = int(dataset.indexes['time'].get_indexer([time], method='nearest')[0])
        if idx < 0:
            raise IndexError(f"No time near {time!r} in the dataset.")
        f = dataset.isel(time=idx)
        if cells is not None:
            f = f.isel(cell=cells)
        return cls(f['x'].values, f['y'].values,
                   f['U'].values[:, 0], f['U'].values[:, 1], f['p'].values,
                   dataset=dataset, index=idx, cells=cells)

    # ------------------------------------------------------------- stepping

    def _at_index(self, idx):
        if self._dataset is None or self._index is None:
            raise NotAssociatedWithDataset("Snapshot is not linked to any dataset")
        n = self._dataset.sizes['time']
        if not 0 <= idx < n:
            raise IndexError(
                f"Time index {idx} out of range for a dataset with {n} snapshots."
            )
        return Snapshot.from_dataset(self._dataset['time'].values[idx],
                                     self._dataset, cells=self._cells)

    def next(self):
        return self._at_index((self._index if self._index is not None else 0) + 1)

    def previous(self):
        return self._at_index((self._index if self._index is not None else 0) - 1)

    @property
    def time(self):
        if self._dataset is None or self._index is None:
            return None
        return float(self._dataset['time'].values[self._index])

    # --------------------------------------------------------------- fields

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def u(self):
        return self._u

    @property
    def v(self):
        return self._v

    @property
    def p(self):
        return self._p

    @property
    def xmin(self):
        return self._x.min()

    @property
    def xmax(self):
        return self._x.max()

    @property
    def ymin(self):
        return self._y.min()

    @property
    def ymax(self):
        return self._y.max()

    @property
    def mean_spacing(self):
        """
        Representative cell size, sqrt(area / n_points).

        The 1-D form (width / n) underestimates this by orders of magnitude on a
        2-D mesh and produced unusable default grids.
        """
        area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        return float(np.sqrt(area / self._x.size))

    # ----------------------------------------------------------- selection

    def mask(self, xmin=None, xmax=None, ymin=None, ymax=None):
        """Boolean per-cell mask for an axis-aligned box. Bounds are optional."""
        return _bounds_mask(self._x, self._y, xmin, xmax, ymin, ymax)

    def mask_rotated(self, xmin=None, xmax=None, ymin=None, ymax=None,
                     angle=0.0, origin=(0, 0), degrees=True):
        """
        Boolean mask for a rectangle rotated by `angle` about `origin`.

        Bounds are in the rotated frame (xmin/xmax streamwise along `angle`).
        """
        return _rotated_bounds_mask(self._x, self._y, xmin, xmax, ymin, ymax,
                                    angle=angle, origin=origin, degrees=degrees)

    def crop(self, xmin=None, xmax=None, ymin=None, ymax=None, mask=None):
        """
        Restrict to a box, with the same signature as Dataset.crop.

        Pass `mask=` instead to use a boolean array you built yourself (e.g.
        from mask_rotated). The dataset link is preserved, so next() and
        previous() return the same region at adjacent times.
        """
        m = self.mask(xmin, xmax, ymin, ymax) if mask is None else np.asarray(mask)
        cells = None
        if self._dataset is not None:
            parent = (np.arange(self._dataset.sizes['cell'])
                      if self._cells is None else np.asarray(self._cells))
            cells = parent[m]
        return Snapshot(self._x[m], self._y[m], self._u[m], self._v[m], self._p[m],
                        dataset=self._dataset, index=self._index, cells=cells)

    # ------------------------------------------------------- interpolation

    @property
    def triangulation(self):
        """Delaunay triangulation of the cell centres, built once and reused."""
        if self._tri is None:
            self._tri = Delaunay(np.column_stack([self._x, self._y]))
        return self._tri

    def cartesian_grid(self, element_size=None):
        """The (x, y) axes that interpolation would use. Includes both bounds."""
        h = self.mean_spacing if element_size is None else float(element_size)
        if h <= 0:
            raise ValueError(f"element_size must be positive, got {element_size!r}")
        nx = int(np.ceil((self.xmax - self.xmin) / h)) + 1
        ny = int(np.ceil((self.ymax - self.ymin) / h)) + 1
        return self.xmin + h * np.arange(nx), self.ymin + h * np.arange(ny)

    def interpolate_on_grid(self, element_size=None):
        """
        Linear interpolation onto a regular grid.

        All three fields share one Delaunay triangulation; building it is the
        expensive part, and griddata rebuilt it per field.
        """
        gx, gy = self.cartesian_grid(element_size=element_size)
        X, Y = np.meshgrid(gx, gy)
        pts = (X, Y)
        tri = self.triangulation
        out = [LinearNDInterpolator(tri, f)(pts) for f in (self._u, self._v, self._p)]
        return CartesianSnapshot(gx, gy, *out)

    def __len__(self):
        return self._x.size

    def __repr__(self):
        t = self.time
        when = f"t={t:g}" if t is not None else "detached"
        return (f"<Snapshot {self._x.size} cells, {when}, "
                f"x[{self.xmin:.3g},{self.xmax:.3g}] "
                f"y[{self.ymin:.3g},{self.ymax:.3g}]>")


class CartesianSnapshot:
    """One instant resampled onto a regular grid, with bilinear sampling."""

    def __init__(self, x, y, u, v, p):
        self._x = np.asarray(x)
        self._y = np.asarray(y)
        shape = (self._y.size, self._x.size)
        for name, f in (("u", u), ("v", v), ("p", p)):
            if np.shape(f) != shape:
                raise ValueError(
                    f"{name} has shape {np.shape(f)}, expected {shape} (len(y), len(x))"
                )
        self._u, self._v, self._p = np.asarray(u), np.asarray(v), np.asarray(p)
        self._element_size = float(self._x[1] - self._x[0]) if self._x.size > 1 else 0.0

        self._interp = {
            name: RegularGridInterpolator((self._y, self._x), f,
                                          bounds_error=False, fill_value=np.nan)
            for name, f in (("u", self._u), ("v", self._v), ("p", self._p))
        }

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def u(self):
        return self._u

    @property
    def v(self):
        return self._v

    @property
    def p(self):
        return self._p

    @property
    def element_size(self):
        return self._element_size

    def sample(self, xq, yq):
        """
        Sample at physical coordinates.

        Returns (u, v, p), NaN outside the grid.
        """
        xq = np.asarray(xq, dtype=float)
        yq = np.asarray(yq, dtype=float)
        pts = np.column_stack([yq.ravel(), xq.ravel()])
        shape = np.broadcast_shapes(xq.shape, yq.shape)
        return tuple(self._interp[n](pts).reshape(shape) for n in ("u", "v", "p"))

    def __repr__(self):
        return (f"<CartesianSnapshot {self._x.size}x{self._y.size} grid, "
                f"h={self._element_size:.4g}>")


# ------------------------------------------------------------------ helpers

def _bounds_mask(x, y, xmin=None, xmax=None, ymin=None, ymax=None):
    mask = np.ones(len(x), dtype=bool)
    if xmin is not None:
        mask &= x >= xmin
    if xmax is not None:
        mask &= x <= xmax
    if ymin is not None:
        mask &= y >= ymin
    if ymax is not None:
        mask &= y <= ymax
    return mask


def _rotate(x, y, angle, origin=(0, 0), degrees=True):
    """
    Express points in a frame rotated by `angle` about `origin`.

    A point lying along the direction `angle` maps onto the positive rotated-x
    axis, so `angle` should be the direction you want to become "downstream"
    (e.g. the freestream angle).
    """
    th = np.radians(angle) if degrees else angle
    c, s = np.cos(th), np.sin(th)
    dx = np.asarray(x) - origin[0]
    dy = np.asarray(y) - origin[1]
    return c * dx + s * dy, -s * dx + c * dy


def _rotate_vector(u, v, angle, degrees=True):
    """Rotate vector components into the same frame as _rotate (no translation)."""
    th = np.radians(angle) if degrees else angle
    c, s = np.cos(th), np.sin(th)
    return c * u + s * v, -s * u + c * v


def _rotated_bounds_mask(x, y, xmin=None, xmax=None, ymin=None, ymax=None,
                         angle=0.0, origin=(0, 0), degrees=True):
    xr_, yr_ = _rotate(x, y, angle, origin, degrees)
    return _bounds_mask(xr_, yr_, xmin, xmax, ymin, ymax)
