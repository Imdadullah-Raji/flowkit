"""
flowkit -- post-processing and Lagrangian analysis of 2-D OpenFOAM simulations.

The common entry points:

    from flowkit import Dataset, pod

    ds   = Dataset.from_netcdf("data/re500_aoa30.nc")
    wake = ds.rotate(30.0).crop_relative(-1, 9, -2, 2)
    res  = pod(wake)
"""

from flowkit.dataprocessing import (
    CartesianSnapshot,
    Dataset,
    NotAssociatedWithDataset,
    Snapshot,
)
from flowkit.io import (
    attach_volumes,
    body_length_scale,
    foam_to_netcdf,
    read_cell_volumes,
    read_foamcase,
)
from flowkit.ftle import FTLEResult, double_gyre, flow_map, ftle
from flowkit.lagrangian import FlowField, Particles, advect
from flowkit.pod import MissingCellVolumes, PODResult, pod

__version__ = "0.1.0"

__all__ = [
    "CartesianSnapshot",
    "Dataset",
    "FTLEResult",
    "FlowField",
    "MissingCellVolumes",
    "NotAssociatedWithDataset",
    "PODResult",
    "Particles",
    "Snapshot",
    "advect",
    "attach_volumes",
    "body_length_scale",
    "double_gyre",
    "flow_map",
    "ftle",
    "foam_to_netcdf",
    "pod",
    "read_cell_volumes",
    "read_foamcase",
]
