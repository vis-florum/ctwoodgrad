"""Public package API for ctwoodgrad."""

from . import fibres, geometry, segmentation, visualisations
from .fibres import (
    fibre_sigmas_for_spacing,
    getFCS,
    getFibreAlignment,
    getFibreTensor,
    getFibreTensorForVoxelSize,
    projectDipDir,
)
from .fibre_chunks import FibreTensorChunk, staged_fibre_tensor_chunks
from .geometry import getSampleAxis
from .segmentation import (
    fill_cavities_slicewise_serial,
    fillCavities,
    findEWLW,
    findInterMode,
    get_threshold_slice,
    get_thresholds_slicewise_MT,
    getMaskStats,
    segment_wood_slicewise,
    segment_wood_volumewise,
    threshold_slicewise_MT,
)
from .visualisations import exportToVTK, prepareDirsVTK, processField

__all__ = [
    "fibres",
    "geometry",
    "segmentation",
    "visualisations",
    "exportToVTK",
    "fill_cavities_slicewise_serial",
    "fillCavities",
    "findEWLW",
    "findInterMode",
    "fibre_sigmas_for_spacing",
    "FibreTensorChunk",
    "get_threshold_slice",
    "get_thresholds_slicewise_MT",
    "getFCS",
    "getFibreAlignment",
    "getFibreTensor",
    "getFibreTensorForVoxelSize",
    "getMaskStats",
    "getSampleAxis",
    "prepareDirsVTK",
    "processField",
    "projectDipDir",
    "segment_wood_slicewise",
    "segment_wood_volumewise",
    "staged_fibre_tensor_chunks",
    "threshold_slicewise_MT",
]
