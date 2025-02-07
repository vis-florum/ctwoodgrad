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

def segmentAir(img, max_rho=1500):
    '''Segment air from wood'''
    t_w = findInterMode(img, 100, 500)
    mask_wood = (img >= t_w) & (img <= max_rho)

    return mask_wood, t_w

def getMaskStats(img, mask):
    '''Get statistics of the image in the mask'''
    rhos = img[mask]
    rho_mean = np.mean(rhos)
    rho_median = np.median(rhos)
    rho_std = np.std(rhos)

    return rho_mean, rho_median, rho_std

def getDensity(img):
    dimg = dip.Image(img)
    return dimg

def getFibreTensor(img):
    '''Fibre estimations in the object'''
    try:
        R, T, L = getFCS(img)
        return R, T, L
    except Exception as e:
        logging.error(f"getFibreTensor failed: {e}")
        return None, None, None