import diplib as dip
import numpy as np
import logging

from ctwoodgrad.gradientanalysis import getFCS

    
def findInterMode(img, rho_min, rho_max):
    '''Find the minimum between two modes of the image histogram'''
    logging.debug("Finding inter modes...")

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

# def segmentAir(img, max_rho=1500):
#     '''Segment air from wood'''
#     t_w = findInterMode(img, 100, 500)
#     mask_wood = (img >= t_w) & (img <= max_rho)

#     return mask_wood, t_w

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


def segmentAir(img, max_rho: int = 1500):
    '''Segment air from wood using intermode threshold and fill holes.'''

    t_w = findInterMode(img, 100, 500)
    logging.info(f"Intermode threshold: {t_w:.2f}")

    # Threshold using DIPLib
    img_dip = dip.Image(img)
    mask = dip.RangeThreshold(img_dip, lowerBound=t_w, upperBound=max_rho)

    # Separate and remove small objects
    mask = dip.Opening(mask)
    label = dip.Label(mask, mode="largest")
    mask = dip.FixedThreshold(label, 1)

    # Fill holes
    # mask = dip.FillHoles(mask, connectivity=2)
    mask = fillCavities(mask)

    return np.asarray(mask), t_w

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

def getFibreTensor(img, sigma=.7, omega=1.5):
    '''Fibre estimations in the object'''
    try:
        R, T, L = getFCS(img, sigma=sigma, omega=omega)
        return R, T, L
    except Exception as e:
        logging.error(f"getFibreTensor failed: {e}")
        return None, None, None