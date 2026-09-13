import numpy as np
import xarray as xr
import pathlib
from scipy.interpolate import griddata, RegularGridInterpolator
from src.io import read_foamcase, body_length_scale

class InvalidSnapshotFormat(Exception):
    pass
class NotAssociatedWithDataset(Exception):
    pass

class Dataset:   
    
    def __init__(self, data: xr.Dataset, length_scale: float = None):
        self._data = data
        self._tri = None
        self.length_scale = length_scale
        
    @property
    def data(self):
        return self._data
    @property
    def times(self):
        return self._data['time'].values

    def snapshot(self, time):
        return Snapshot.from_dataset(time, self._data)
    def snapshots(self, times= None):
        times = self.time if times is None else times 
        for t in times:
            yield  self.snapshot(t)

    @classmethod
    def from_foam(cls, casepath, patch='airfoil'):
        data = read_foamcase(casepath, patch=patch)                       # pure I/O
        L = body_length_scale(casepath, patch=patch)      # computed once, at load time
        return cls(data, length_scale=L)
    
    @classmethod
    def from_dataset(cls, dataset:xr.Dataset):
        return cls(dataset, dataset.attrs.get('length_scale'))
    
    @classmethod
    def from_netcdf(cls, source):
        ds = xr.open_dataset(source)
        len_scale= ds.attrs.get('length_scale')
        return(cls(ds, len_scale))

    def crop(self, xmin=None, xmax=None, ymin=None, ymax=None) -> "Dataset":
        x = self._data['x'].values
        y = self._data['y'].values
        mask = _bounds_mask(x, y, xmin, xmax, ymin, ymax)
        return Dataset(self._data.isel(cell=mask), length_scale=self.length_scale)

    def crop_relative(self, xmin, xmax, ymin, ymax, length_scale=None, origin=(0, 0)) -> "Dataset":
        L = length_scale if length_scale is not None else self.length_scale
        if L is None:
            raise ValueError(
                "No length_scale given and none stored on this Dataset "
                "(construct via from_foam, or pass length_scale= explicitly)."
            )
        x0, y0 = origin
        return self.crop(
            xmin=x0 + xmin * L, xmax=x0 + xmax * L,
            ymin=y0 + ymin * L, ymax=y0 + ymax * L,
        )

    def crop_rotated(self, xmin=None, xmax=None, ymin=None, ymax=None,
                     angle=0.0, origin=(0, 0), degrees=True) -> "Dataset":
        """
        Crop to a rectangle whose axes are rotated by `angle` about `origin`.

        Bounds are given in the rotated frame: xmin/xmax run along `angle`
        (streamwise), ymin/ymax across it. Coordinates are left in the lab
        frame -- only the cell selection changes. Use .rotate() as well if you
        want the region itself expressed in the rotated frame.
        """
        x = self._data['x'].values
        y = self._data['y'].values
        mask = _rotated_bounds_mask(x, y, xmin, xmax, ymin, ymax,
                                    angle=angle, origin=origin, degrees=degrees)
        return type(self)(self._data.isel(cell=mask), length_scale=self.length_scale)

    def crop_rotated_relative(self, xmin, xmax, ymin, ymax, angle=0.0,
                              length_scale=None, origin=(0, 0), degrees=True) -> "Dataset":
        """As crop_rotated, with bounds in multiples of the body length scale."""
        L = length_scale if length_scale is not None else self.length_scale
        if L is None:
            raise ValueError(
                "No length_scale given and none stored on this Dataset "
                "(construct via from_foam, or pass length_scale= explicitly)."
            )
        return self.crop_rotated(
            xmin=xmin * L, xmax=xmax * L, ymin=ymin * L, ymax=ymax * L,
            angle=angle, origin=origin, degrees=degrees,
        )

    def rotate(self, angle, origin=(0, 0), degrees=True) -> "Dataset":
        """
        Re-express the whole dataset in a frame rotated by `angle` about `origin`.

        Rotates the x/y coordinates *and* the U components, so the data stays
        physically consistent and particle advection remains correct. After
        this, a region aligned with `angle` is axis-aligned, so plain crop()
        and interpolateOnCartesianGrid() work on it without waste.
        """
        ds = self._data.copy()
        xr_, yr_ = _rotate(ds['x'].values, ds['y'].values, angle, origin, degrees)

        U = np.array(ds['U'].values)
        U[..., 0], U[..., 1] = _rotate_vector(U[..., 0], U[..., 1], angle, degrees)

        ds = ds.assign_coords(x=('cell', xr_), y=('cell', yr_))
        ds['U'] = (self._data['U'].dims, U)
        ds.attrs['frame_angle'] = angle if degrees else np.degrees(angle)
        ds.attrs['frame_origin'] = tuple(origin)
        return type(self)(ds, length_scale=self.length_scale)

class Snapshot:

    def __init__(self, x, y, p, u, v, dataset= None, index= None):
        self._x = x
        self._y = y
        self._p = p 
        self._u = u
        self._v = v
        self._dataset = dataset
        self._index= index

        self._xmin = np.min(x)
        self._ymin = np.min(y)
        self._xmax = np.max(x)
        self._ymax = np.max(y)
        self._numpoints = len(x)

        self._meanspacing = (self._xmax-self._xmin)/self._numpoints

    @classmethod
    def from_xarray(cls, f:xr.DataArray):
        x = f['x'].values
        y = f['y'].values
        p = f['p'].values
        u = f['U'].values[:,0]
        v = f['U'].values[:,1]

        return cls( x, y , p, u, v)

    @classmethod
    def from_dataset(cls, time, dataset:xr.Dataset):

        f = dataset.sel(time = time, method= 'nearest')

        x = f['x'].values
        y = f['y'].values
        p = f['p'].values
        u = f['U'].values[:,0]
        v = f['U'].values[:,1]

        idx = dataset.indexes["time"].get_loc(time)

        return cls(x,y,p,u,v, dataset = dataset, index =idx )
    
    # TODO Need to handle index out of bound cases

    def next(self):
        if self._index is None:
            raise NotAssociatedWithDataset(
                "Snapshot is not linked to any dataset"
            )
        idx = self._index+1
        time = self._dataset.time[idx].item()
        return Snapshot.from_dataset(time, self._dataset)
    def previous(self):
        if self._index is None:
            raise NotAssociatedWithDataset(
                "Snapshot is not linked to any dataset"
            )
        idx = self._index -1
        time = self._dataset.time[idx].item()
        return Snapshot.from_dataset(time, self._dataset)


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
        return self._xmin
    @property
    def xmax(self):
        return self._xmax
    @property
    def ymin(self):
        return self._ymin
    @property
    def ymax(self):
        return self._ymax
    
    def crop(self, mask):
        return Snapshot(
            self._x[mask],
            self._y[mask],
            self._p[mask],
            self._u[mask],
            self._v[mask],
        )

        
    def mask(self, xmin=None, xmax=None, ymin=None, ymax=None):
        valid = (
            (self._x>=xmin) & 
            (self._x<=xmax) & 
            (self._y>=ymin) & 
            (self._y<=ymax)
        )
        #add None logic later
        return valid

    def mask_rotated(self, xmin=None, xmax=None, ymin=None, ymax=None,
                     angle=0.0, origin=(0, 0), degrees=True):
        """
        Boolean mask for a rectangle rotated by `angle` about `origin`.

        Bounds are in the rotated frame (xmin/xmax streamwise along `angle`).
        Unlike mask(), omitted bounds are simply not applied.
        """
        return _rotated_bounds_mask(self._x, self._y, xmin, xmax, ymin, ymax,
                                    angle=angle, origin=origin, degrees=degrees)



    def interpolateOnCartesianGrid(self, element_size=None ):
        x,y = self.constructCartesianGrid(element_size=element_size)
        X,Y = np.meshgrid(x,y)

        p_interp = griddata((self._x, self._y), self._p, (X,Y), method = 'linear')
        u_interp = griddata((self._x, self._y), self._u, (X,Y), method = 'linear')
        v_interp = griddata((self._x, self._y), self._v, (X,Y), method = 'linear')

        return CartesianSnapshot(x,y, p_interp, u_interp, v_interp)

    def constructCartesianGrid(self, element_size=None):
        if element_size==None:
            element_size = self._meanspacing
        return np.arange(self._xmin, self._xmax, element_size), np.arange(self._ymin, self._ymax, element_size)
        

class CartesianSnapshot:
    def __init__(self, x, y, p, u, v):

        if p.shape != (len(y), len(x)):
            raise ValueError("p has incorrect shape")

        if u.shape != (len(y), len(x)):
            raise ValueError("u has incorrect shape")

        if v.shape != (len(y), len(x)):
            raise ValueError("v has incorrect shape")

        self._x = x
        self._y = y
        self._elementsize= x[1]-x[0]

        self._p = p
        self._u = u
        self._v = v

        self._p_interp = RegularGridInterpolator(
            (y, x), p,
            bounds_error=False,
            fill_value=np.nan
        )

        self._u_interp = RegularGridInterpolator(
            (y, x), u,
            bounds_error=False,
            fill_value=np.nan
        )

        self._v_interp = RegularGridInterpolator(
            (y, x), v,
            bounds_error=False,
            fill_value=np.nan
        )

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def p(self):
        return self._p

    @property
    def u(self):
        return self._u

    @property
    def v(self):
        return self._v
    @property
    def elementSize(self):
        return self._elementsize

    def sample(self, xq, yq):
        """
        Sample field at physical coordinates.

        Parameters
        ----------
        xq, yq : float or ndarray

        Returns
        -------
        u, v, p
        """

        pts = np.column_stack([
            np.asarray(yq).ravel(),
            np.asarray(xq).ravel()
        ])

        u = self._u_interp(pts)
        v = self._v_interp(pts)
        p = self._p_interp(pts)

        shape = np.broadcast(xq, yq).shape

        return (
            u.reshape(shape),
            v.reshape(shape),
            p.reshape(shape)
        )


def _rotate(x, y, angle, origin=(0, 0), degrees=True):
    """
    Express points in a frame rotated by `angle` about `origin`.

    A point lying along the direction `angle` maps onto the positive
    rotated-x axis, so `angle` should be the direction you want to become
    "downstream" (e.g. the freestream angle).
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