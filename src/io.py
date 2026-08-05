from fluidfoam import readmesh, readscalar, readvector
from pathlib import Path
import xarray as xr
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


def read_foamcase(casepath:str, patch= 'airfoil'):

    case = Path(casepath)

    times = sorted( (float(d.name), d.name) for d in case.iterdir()
            if d.is_dir() and d.name[0].isdigit())
    x, y, z = readmesh(casepath, verbose = False)
    n = x.size
    fields = {"p":readscalar, "U":readvector}
    data = {f: [] for f in fields}
    for _, tdir in times:
        for f, reader in fields.items():
            a = np.asarray(reader(str(case), tdir, f, verbose=False))
            if a.shape[-1] == 1:                       # uniform -> broadcast
                a = np.broadcast_to(a, (*a.shape[:-1], n))
            data[f].append(a.T if a.ndim > 1 else a)   # (3,n)
    ds = xr.Dataset(
        {f: (("time", "cell", "comp")[:np.stack(v).ndim], np.stack(v))
        for f, v in data.items()},
        coords={"time": [t for t, _ in times], "x": ("cell", x), "y": ("cell", y)},
    )
    ds.attrs['length_scale'] = body_length_scale(casepath=casepath, patch=patch)
    ds.attrs['source_case']= casepath
    
    return ds
def foam_to_netcdf(casepath, outpath, patch= 'airfoil'):
    ds = read_foamcase(casepath=casepath, patch=patch)
    ds.to_netcdf(outpath)
    return outpath

def body_length_scale(casepath, patch='airfoil'):
    x, y, z = readmesh(casepath, boundary=patch, verbose=False)
    pts = np.column_stack([x, y])
    hull_pts = pts[ConvexHull(pts).vertices]   # only hull vertices matter for max distance
    return pdist(hull_pts).max()

