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