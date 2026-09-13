"""
Where the cases live.

Scripts used to open with three absolute paths each, and several of them still
pointed at a directory that had been renamed out from under them. Register a
case once here and refer to it by name instead.

    from flowkit.cases import case

    c    = case("re500_aoa30")
    ds   = c.dataset()                       # NetCDF + cell volumes, ready to use
    wake = ds.rotate(c.aoa).crop_relative(-1, 9, -2, 2)
    fig.savefig(c.figure("pod_modes.png"))

Path resolution for the NetCDF, in order:
  1. an explicit `nc=` on the registry entry
  2. $FLOWKIT_DATA/<name>.nc
  3. <repo>/data/<name>.nc
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FLOWKIT_DATA", REPO / "data"))
FIGURES = Path(os.environ.get("FLOWKIT_FIGURES", REPO / "figures"))


@dataclass(frozen=True)
class Case:
    """One simulation: where it is, what it contains, how to load it."""

    name: str
    path: Path                      # the OpenFOAM case directory
    patch: str = "airfoil"          # boundary patch used for the length scale
    u_inf: tuple = (1.0, 0.0)       # freestream, sets the wake axis
    nc: Path = None                 # explicit NetCDF path; otherwise resolved
    figures: Path = None
    notes: str = ""

    @property
    def netcdf(self) -> Path:
        if self.nc is not None:
            return Path(self.nc)
        env = DATA / f"{self.name}.nc"
        return env if env.exists() else REPO / "data" / f"{self.name}.nc"

    @property
    def aoa(self) -> float:
        """Freestream angle in degrees -- the angle to rotate the wake onto +x."""
        return float(np.degrees(np.arctan2(self.u_inf[1], self.u_inf[0])))

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.u_inf))

    def figure(self, filename) -> Path:
        d = Path(self.figures) if self.figures is not None else FIGURES
        d.mkdir(parents=True, exist_ok=True)
        return d / filename

    def dataset(self, volumes=True):
        """
        Load the cached NetCDF as a Dataset, with cell volumes attached.

        Volumes live on the case rather than in the NetCDF for datasets that
        were converted before flowkit read them, so this re-attaches from the
        case directory instead of forcing a reconversion.
        """
        from flowkit.dataprocessing import Dataset
        from flowkit.io import attach_volumes

        nc = self.netcdf
        if not nc.exists():
            raise FileNotFoundError(
                f"No NetCDF for case {self.name!r} at {nc}.\n"
                f"Convert it first:\n"
                f"    from flowkit.io import foam_to_netcdf\n"
                f"    foam_to_netcdf({str(self.path)!r}, {str(nc)!r}, "
                f"patch={self.patch!r})\n"
                f"or point $FLOWKIT_DATA at the directory that holds it."
            )
        ds = Dataset.from_netcdf(nc)
        if volumes and "V" not in ds.data.coords:
            ds = Dataset(attach_volumes(ds.data, str(self.path)),
                         length_scale=ds.length_scale)
        return ds

    def convert(self, overwrite=False) -> Path:
        """Read the OpenFOAM case and write the NetCDF cache."""
        from flowkit.io import foam_to_netcdf

        nc = self.netcdf
        if nc.exists() and not overwrite:
            return nc
        nc.parent.mkdir(parents=True, exist_ok=True)
        return Path(foam_to_netcdf(str(self.path), str(nc), patch=self.patch))


_SIN30, _COS30 = 0.500000000, 0.866025404

REGISTRY = {
    "re500_aoa30": Case(
        name="re500_aoa30",
        path=Path("/home/raji/Research/Thesis/static_airfoil/re500_aoa30"),
        patch="airfoil",
        u_inf=(_COS30, _SIN30),
        # Converted before the repo had a data/ directory. Drop this line after
        # moving the file into <repo>/data/ (or set $FLOWKIT_DATA).
        nc=Path("/home/raji/Research/analyze_cfd/data/re500_aoa30.nc"),
        notes="Static airfoil, Re=500, AoA=30. Static 2-D mesh, 63360 cells, "
              "chord 1.0, times 120.5-245.65. Sheds at St~0.319.",
    ),
}


def case(name) -> Case:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown case {name!r}. Registered: {sorted(REGISTRY)}.\n"
            f"Add it to REGISTRY in flowkit/cases.py."
        )
    return REGISTRY[name]


def cases():
    return sorted(REGISTRY)
