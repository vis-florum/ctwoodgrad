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
    mask = dip.FillHoles(mask, connectivity=2)

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

def getFibreTensor(img):
    '''Fibre estimations in the object'''
    try:
        R, T, L = getFCS(img)
        return R, T, L
    except Exception as e:
        logging.error(f"getFibreTensor failed: {e}")
        return None, None, None