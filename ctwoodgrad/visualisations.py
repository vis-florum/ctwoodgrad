import diplib as dip
import numpy as np
from pyevtk.hl import imageToVTK
import logging


def prepareDirsVTK(dip_vecs):
    """Convert a DIPlib vector image into a (fx, fy, fz) Fortran-style tuple."""
    if dip.AreDimensionsReversed(): # this is standard order in diplib
        fx = np.asfortranarray(dip_vecs(2))
        fy = np.asfortranarray(dip_vecs(1))
        fz = np.asfortranarray(dip_vecs(0))
    else:
        fx = np.asfortranarray(dip_vecs(0))
        fy = np.asfortranarray(dip_vecs(1))
        fz = np.asfortranarray(dip_vecs(2))
    
    return (fx, fy, fz)


def processField(value):
    """Prepare data field based on whether it's a DIPlib image or NumPy array."""
    if isinstance(value, dip.Image):
        if value.TensorElements() == 3:
            # It's a vector field
            return prepareDirsVTK(value)
        elif value.TensorElements() == 1:
            # Scalar image
            return np.asfortranarray(value)
        else:
            logging.error(f"Unsupported number of tensor elements for VTK export: {value.TensorElements()}")
            raise ValueError(f"Unsupported number of tensor elements for VTK export: {value.TensorElements()}")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.integer):
            return value.astype(np.uint16)
        else:
            return value
    else:
        logging.error(f"Unsupported type for VTK export: {type(value)}")
        raise TypeError(f"Unsupported type for VTK export: {type(value)}")
   

def exportToVTK(outfile, **kwargs):
    """
    General-purpose VTK exporter. Automatically handles dip.Image and numpy arrays.
    Converts:
      - DIPlib vector images to VTK vector fields
      - DIPlib scalar images to Fortran-style arrays
      - int numpy arrays to uint16
      - other numpy arrays left as-is
    """
    pointData = {}
    for key, val in kwargs.items():
        pointData[key] = processField(val)

    imageToVTK(outfile, pointData=pointData)
    return