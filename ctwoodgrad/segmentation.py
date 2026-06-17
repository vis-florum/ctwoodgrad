import diplib as dip
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import logging


MAX_WORKERS = 8
LOWER_BOUND = 100
MODEBOUND = 200
LATEWOOD = 500
UPPER_BOUND = 1500


### BASIC FUNCTIONS

def _normalize_slice_axis(slice_axis, ndim):
    """Return a positive axis index, or fail before work starts."""
    if ndim != 3:
        raise ValueError("Slicewise segmentation expects a 3D numpy array.")

    try:
        axis = int(slice_axis)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "slice_axis must be an integer axis: 0, 1, 2, or a negative equivalent."
        ) from exc

    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"slice_axis={slice_axis} is outside the valid axes for shape with {ndim} dimensions."
        )

    return axis


def _as_slices_first(arr, slice_axis):
    """
    Put the caller's slice axis first and make each 2D plane contiguous.

    Worker threads then walk memory linearly inside each slice. The caller never
    sees this internal layout; results are restored before return.
    """
    axis = _normalize_slice_axis(slice_axis, arr.ndim)
    return np.ascontiguousarray(np.moveaxis(arr, axis, 0)), axis


def _restore_slice_axis(arr, slice_axis):
    """Move a slices-first result back to the caller's original axis order."""
    return np.ascontiguousarray(np.moveaxis(arr, 0, slice_axis))


def findInterMode(img, rho_min, rho_max):
    '''Find the minimum between two modes of the image histogram'''
    # logging.debug("Finding inter modes...")

    mask = (img >= rho_min) & (img <= rho_max)
    if not np.any(mask):  # Avoid errors if no pixels in range
        logging.warning("No valid pixels found in range.")
        return None

    rhos = img[mask]
    nbins = rho_max - rho_min + 1
    freq, bin_edges = np.histogram(rhos, bins=nbins)
    freq = dip.Gauss(freq, 3)

    histmin = np.argmin(freq)
    t = bin_edges[histmin]

    return t


# Multithreading is worthwile for thresholding, as each slice is independent and the operation is relatively expensive.
def get_threshold_slice(args):
    k, slice_np, lower, latewood = args
    t = findInterMode(slice_np, lower, latewood)
    if t is None:
        t = np.nan
    return k, t


def get_thresholds_slicewise_MT(img_np, lower, latewood, max_workers=None, slice_axis=-1):
    """
    Compute one threshold per slice.
    Pass the axis that indexes slices.
    """
    img_slices, _ = _as_slices_first(np.asarray(img_np), slice_axis)
    sz = img_slices.shape[0]
    ts = np.empty(sz, dtype=np.float64)

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)

    # Send one contiguous 2D slice per task; return only the scalar threshold.
    jobs = ((k, img_slices[k], lower, latewood) for k in range(sz))
    logging.debug(f"Starting multithreaded thresholding with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for k, t in ex.map(get_threshold_slice, jobs):
            ts[k] = t

    return ts


def threshold_slicewise_MT(
    img_np,
    lower=LOWER_BOUND,
    intermode_limit=MODEBOUND,
    latewood=LATEWOOD,
    upper=UPPER_BOUND,
    max_workers=None,
    slice_axis=-1,
):
    """
    Threshold each 2D slice and return a mask in the input axis order.

    slice_axis tells the function where the slice stack lives. Use 0 for
    SimpleITK-style arrays shaped as (z, y, x); use -1 for (..., z).
    """
    img_slices, axis = _as_slices_first(np.asarray(img_np), slice_axis)
    ts = get_thresholds_slicewise_MT(
        img_slices,
        lower,
        latewood,
        max_workers=max_workers,
        slice_axis=0,
    )
    ts[(ts > intermode_limit) | np.isnan(ts)] = lower   # Protect weak edge slices.

    # Threshold in slices-first order for fast broadcasting, then restore shape.
    thresh = ts[:, None, None]
    mask_slices = (img_slices >= thresh) & (img_slices <= upper)
    mask = _restore_slice_axis(mask_slices, axis)
    return mask, ts


def fillCavities(mask):
    '''
    Bit more general than filling holes, as it also fills cavities
    that are not connected to the background.
    '''
    
    L_object = dip.Label(mask, mode="largest")    # solid label is 1
    B_object = L_object > 0

    L_internalobjects = dip.Label(~B_object, boundaryCondition=["remove"])   # Remove border-touching objects

    # Map the labels to 1 + label_nr (shift all by 1)
    labels_internalobjects = dip.ListObjectLabels(L_internalobjects)    # BG 0 i not included
    # Return upon empty list of internal objects (no pores)
    if len(labels_internalobjects) == 0:
        return mask
    lut = np.zeros(len(labels_internalobjects) + 1, dtype=np.uint32)
    lut[1:] = np.array(labels_internalobjects) + 1
    L_internalobjects = dip.LookupTable(lut).Apply(L_internalobjects)
    assert len(dip.ListObjectLabels(L_internalobjects)) == len(lut) - 1
    assert np.min(dip.ListObjectLabels(L_internalobjects)) == 2 

    L_object_and_internals = L_object | L_internalobjects  # solid is label 1, pores start at label 2

    # Fill pores using Graph Representation of the regions
    # If edge weights are large, then the relative areag connecting the regions is small
    G_solid_and_internals = dip.RegionAdjacencyGraph(L_object_and_internals, mode="touching")    # region with ID 0 (the background) is not included in the graph
    MSF_solid_and_internals = G_solid_and_internals.MinimumSpanningForest([1])  # Only the regions touching label 1

    L_closed = dip.Relabel(L_object_and_internals,MSF_solid_and_internals)
    B_closed = L_closed > 0
    return B_closed


# Multithreading is not worthwile for filling cavities
def fill_cavities_slicewise_serial(mask_np, slice_axis=-1):
    """Fill cavities per 2D slice, preserving the input axis order."""
    mask_slices, axis = _as_slices_first(np.asarray(mask_np, dtype=bool), slice_axis)
    mask_filled = np.empty_like(mask_slices, dtype=bool)

    for k in range(mask_slices.shape[0]):
        mask_filled[k] = np.asarray(fillCavities(mask_slices[k]), dtype=bool)

    return _restore_slice_axis(mask_filled, axis)



#############################################
### COMPOSITE FUNCTIONS

# def segmentAir(img, max_rho=1500):
#     '''Segment air from wood'''
#     t_w = findInterMode(img, 100, 500)
#     mask_wood = (img >= t_w) & (img <= max_rho)

#     return mask_wood, t_w

def segment_wood_volumewise(img, upper: int = UPPER_BOUND):
    '''Segment air from wood using intermode threshold and fill holes.'''

    t_w = findInterMode(img, 100, 500)
    logging.info(f"Intermode threshold: {t_w:.2f}")

    # Threshold using DIPLib
    img_dip = dip.Image(img)
    mask = dip.RangeThreshold(img_dip, lowerBound=t_w, upperBound=upper)

    # Separate and remove small objects
    mask = dip.Opening(mask)
    label = dip.Label(mask, mode="largest")
    mask = dip.FixedThreshold(label, 1)

    # Fill holes
    # mask = dip.FillHoles(mask, connectivity=2)
    mask = fillCavities(mask)

    return np.asarray(mask), t_w


def segment_wood_slicewise(
    img_np,
    fill_slicewise: bool = True,
    lower=LOWER_BOUND,
    intermode_limit=MODEBOUND,
    latewood=LATEWOOD,
    upper=UPPER_BOUND,
    max_workers=MAX_WORKERS,
    slice_axis=-1,
):
    """
    Segment wood slice by slice.

    The caller must state which axis indexes slices. Use slice_axis=0 for
    SimpleITK numpy arrays and slice_axis=-1 for the old last-axis convention.
    """
    M_np, ts = threshold_slicewise_MT(
        img_np,
        lower=lower,
        intermode_limit=intermode_limit,
        latewood=latewood,
        upper=upper,
        max_workers=max_workers,
        slice_axis=slice_axis,
    )

    M_dip = dip.Image(M_np)
    M_dip = dip.Opening(M_dip)
    label = dip.Label(M_dip, mode="largest")
    M_dip = label > 0
    M_np = np.asarray(M_dip, dtype=bool)

    if fill_slicewise:
        M_np_filled = fill_cavities_slicewise_serial(M_np, slice_axis=slice_axis)
    else:
        M_np_filled = np.asarray(fillCavities(M_np), dtype=bool)
    
    return M_np_filled, ts


# def getMaskStats(img, mask):
#     '''Get statistics of the image in the mask'''
#     rhos = img[mask]
#     rho_mean = np.mean(rhos)
#     rho_median = np.median(rhos)
#     rho_std = np.std(rhos)

#     return rho_mean, rho_median, rho_std

def getMaskStats(img, mask):
    """
    Compute statistics of the image within the specified mask using DIPlib.
    """
    # Convert numpy arrays to DIPlib images
    img_dip = dip.Image(img)
    mask_dip = dip.Image(mask)

    # Compute statistics within the mask
    stats = dip.SampleStatistics(img_dip, mask_dip)
    mean = stats.mean
    std_dev = stats.standardDev

    median = dip.Median(img_dip, mask=mask_dip)[0][0]

    return mean, median, std_dev


def findEWLW(img, mask_wood):
    '''Find the EW / LW content of the wood mask using curvature analysis'''
    imgn = img - np.min(img)
    imgn = imgn / np.max(imgn)
    
    logging.debug("Building Hessian... ")
    H = dip.Hessian(imgn)
    logging.debug(f"    {H}")

    logging.debug("Eigen decomposition... ")
    eigenvalues, eigenvectors = dip.EigenDecomposition(H)
    logging.debug(f"    {eigenvalues}... ")
    logging.debug(f"    {eigenvectors}... ")

    # Hessian Energy
    logging.debug("Get Laplacian... ")
    H_E = dip.Trace(eigenvalues)
    H_E = np.array(H_E)

    # EW / LW content
    logging.debug("Masking... ")
    mask_ew = mask_wood & (H_E < 0)
    mask_lw = mask_wood & (H_E >= 0)
    
    # Ratios
    logging.debug("EW/LW rations... ")
    r_ew = np.sum(mask_ew) / np.sum(mask_wood)
    r_lw = np.sum(mask_lw) / np.sum(mask_wood)
    # check if r_ew + r_lw is approximately 1
    if abs(r_ew + r_lw - 1) > 0.01:
        logging.error("Warning: r_ew + r_lw = " + str(r_ew + r_lw) + ", instead of 1!")

    # Create a labael map where 0=background, 1=EW, 2=LW
    logging.debug("Creating label map... ")
    labels_ewlw = np.zeros_like(img)
    labels_ewlw[mask_ew] = 1
    labels_ewlw[mask_lw] = 2
    
    return labels_ewlw, r_ew
