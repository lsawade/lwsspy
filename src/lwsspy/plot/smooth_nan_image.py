import numpy as np
import scipy as sp


def smooth_nan_image(image, sigma: float, truncate: float):
    """Smoothes images that contains nans

    Parameters
    ----------
    image : array_like
        matrix with nans
    sigma : float
        standard deviation of the Gaussian kernel
    truncate : float
        truncate filter at this many sigmas

    Returns
    -------
    array_like
        smoothed image still containing the nans in the same
        places

    Notes
    -----

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.07 23.00


    """

    nanpos = np.isnan(image)
    sigma = 1.25       # standard deviation for Gaussian kernel
    truncate = 2.0    # truncate filter at this many sigmas
    V = image.copy()
    V[nanpos] = 0
    VV = sp.ndimage.gaussian_filter(
        V, sigma=sigma, truncate=truncate)

    W = 0*image.copy()+1
    W[nanpos] = 0
    WW = sp.ndimage.gaussian_filter(
        W, sigma=sigma, truncate=truncate)

    # Makes the division computable
    epsilon = 0.0000001
    Z = VV/(WW+epsilon)
    Z[nanpos] = np.nan

    return Z
