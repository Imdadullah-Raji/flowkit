from fluidfoam import readmesh, readscalar, readvector
from pathlib import Path
import warnings
import xarray as xr
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


def _numeric_time_dirs(case: Path):
    """Sorted (value, name) pairs for time directories whose name parses as a float."""
    times = []
    for d in case.iterdir():
        if not (d.is_dir() and d.name[0].isdigit()):
            continue
        try:
            times.append((float(d.name), d.name))
        except ValueError:
            continue          # e.g. 0.orig, 0_backup
    return sorted(times)


def read_foamcase(casepath:str, patch= 'airfoil'):

    case = Path(casepath)

    times = _numeric_time_dirs(case)
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
    coords = {"time": [t for t, _ in times], "x": ("cell", x), "y": ("cell", y)}
    # Cell volumes are optional -- only POD needs them -- so a case without them
    # still converts. src.pod raises a pointed error if they turn out to be missing.
    try:
        coords["V"] = ("cell", read_cell_volumes(casepath))
    except FileNotFoundError as e:
        warnings.warn(f"{e}\n(continuing without cell volumes)", stacklevel=2)

    ds = xr.Dataset(
        {f: (("time", "cell", "comp")[:np.stack(v).ndim], np.stack(v))
        for f, v in data.items()},
        coords=coords,
    )
    ds.attrs['length_scale'] = body_length_scale(casepath=casepath, patch=patch)
    ds.attrs['source_case']= casepath
    
    return ds


def foam_to_netcdf(casepath, outpath, patch= 'airfoil'):
    ds = read_foamcase(casepath=casepath, patch=patch)
    ds.to_netcdf(outpath)
    return outpath

_VOLUME_NAMES = ("V", "Vc")   # OpenFOAM.org dev writes "Vc"; older/renamed cases use "V"

_NO_VOLUMES = (
    "No cell-volume field found in {case}.\n"
    "Write one (static mesh -> keep a single copy in constant/):\n"
    "    cd {case}\n"
    "    postProcess -func writeCellVolumes -latestTime\n"
    "    mv <latestTime>/Vc constant/V\n"
    "The -latestTime is important: without it the field is written into every "
    "time directory."
)


def read_cell_volumes(casepath):
    """
    Cell volumes for a case, as a (n_cell,) array aligned with readmesh ordering.

    Looks in constant/ first (correct for a static mesh), then the latest numeric
    time directory. Accepts either the OpenFOAM 'Vc' name or a renamed 'V'.
    """
    case = Path(casepath)
    times = _numeric_time_dirs(case)
    sources = ["constant"] + ([times[-1][1]] if times else [])

    for tdir in sources:
        for name in _VOLUME_NAMES:
            if not (case / tdir / name).is_file():
                continue
            return np.asarray(readscalar(str(case), tdir, name, verbose=False))

    raise FileNotFoundError(_NO_VOLUMES.format(case=casepath))


def attach_volumes(ds: xr.Dataset, casepath=None) -> xr.Dataset:
    """
    Add the 'V' cell coordinate to a dataset that was written before volumes
    were read (e.g. an existing NetCDF), without reconverting the case.

    Falls back to the dataset's own 'source_case' attribute.
    """
    casepath = casepath or ds.attrs.get("source_case")
    if casepath is None:
        raise ValueError(
            "No casepath given and the dataset has no 'source_case' attribute."
        )
    V = read_cell_volumes(casepath)
    if V.size != ds.sizes["cell"]:
        raise ValueError(
            f"Case has {V.size} cells but the dataset has {ds.sizes['cell']}. "
            "The dataset was probably cropped -- attach volumes before cropping."
        )
    return ds.assign_coords(V=("cell", V))


def body_length_scale(casepath, patch='airfoil'):
    x, y, z = readmesh(casepath, boundary=patch, verbose=False)
    pts = np.column_stack([x, y])
    hull_pts = pts[ConvexHull(pts).vertices]   # only hull vertices matter for max distance
    return pdist(hull_pts).max()

