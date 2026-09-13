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

Cases are declared in `cases.toml` -- read from $FLOWKIT_CASES if set, else
from <repo>/cases.toml. Registering a simulation is a config edit, not a code
edit; nothing about your data lives in the package.

Path resolution for the NetCDF, in order:
  1. an explicit `nc=` on the registry entry
  2. $FLOWKIT_DATA/<name>.nc
  3. <repo>/data/<name>.nc
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:                                    # stdlib from 3.11
    import tomllib
except ModuleNotFoundError:             # pragma: no cover
    import tomli as tomllib

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


CASES_FILE = Path(os.environ.get("FLOWKIT_CASES", REPO / "cases.toml"))

_CACHE = None


def _build(name, entry, defaults):
    merged = {**defaults, **entry}
    if "path" not in merged:
        raise KeyError(f"Case {name!r} has no 'path' key.")
    known = {"path", "patch", "u_inf", "nc", "figures", "notes"}
    unknown = set(merged) - known
    if unknown:
        raise KeyError(
            f"Case {name!r} has unknown key(s) {sorted(unknown)}; "
            f"known keys are {sorted(known)}."
        )
    return Case(
        name=name,
        path=Path(merged["path"]),
        patch=merged.get("patch", "airfoil"),
        u_inf=tuple(merged.get("u_inf", (1.0, 0.0))),
        nc=Path(merged["nc"]) if merged.get("nc") else None,
        figures=Path(merged["figures"]) if merged.get("figures") else None,
        notes=merged.get("notes", "").strip(),
    )


def load_registry(path=None):
    """Parse the case file. Raises if it is missing -- there is no implicit empty."""
    path = Path(path) if path is not None else CASES_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"No case file at {path}.\n"
            f"Create it, or point $FLOWKIT_CASES at one. Minimal form:\n\n"
            f"    [cases.my_case]\n"
            f'    path  = "/path/to/openfoam/case"\n'
            f'    patch = "airfoil"\n'
            f"    u_inf = [1.0, 0.0]\n"
        )
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    defaults = doc.get("defaults", {})
    return {n: _build(n, e, defaults) for n, e in doc.get("cases", {}).items()}


def registry(reload=False):
    """The parsed registry, cached. `reload=True` re-reads the file."""
    global _CACHE
    if _CACHE is None or reload:
        _CACHE = load_registry()
    return _CACHE


def case(name) -> Case:
    reg = registry()
    if name not in reg:
        raise KeyError(
            f"Unknown case {name!r}. Registered: {sorted(reg)}.\n"
            f"Add a [cases.{name}] table to {CASES_FILE}."
        )
    return reg[name]


def cases():
    return sorted(registry())
