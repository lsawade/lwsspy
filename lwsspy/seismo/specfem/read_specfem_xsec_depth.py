from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import sys
import cartopy

import lwsspy as lpy


def read_specfem_xsec_depth(filename: str, res: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reads in a specfem generated cross section and outputs a gridded version
    of it to map it out.

    Parameters
    ----------
    filename : str
        Filename
    res : float, optional
        Resolution for the output grid to plot, by default 0.25


    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray]
        Output Tuple - llon, llat, radius, value, perturb, difference, distance

    """

    # Read file using numpy's load txt
    array = np.loadtxt(filename, dtype=float, comments='#', delimiter=None,
                       converters=None, skiprows=0, usecols=None,
                       unpack=False, ndmin=0, encoding='bytes',
                       max_rows=None)

    # Split params
    lon = array[:, 0]  # Longitude
    lat = array[:, 1]  # Latitude
    rad = array[:, 2]  # Radius
    val = array[:, 3]  # Value of the crossection parameter
    per = array[:, 4]  # perturbation wrt. average value
    dif = array[:, 5]  # difference from average val
    dis = array[:, 6]  # distance to closest point in mesh

    # Create desired grid for plotting
    llon, llat = np.meshgrid(np.arange(-180.0, 180.0 + res, res),
                             np.arange(-90.0, 90.0 + res, res))

    # Interpolate the map
    SNN = lpy.SphericalNN(lat, lon)  # Creating a kdtree, and use it to interp
    rad = SNN.interp(rad, llat, llon, no_weighting=True)
    val = SNN.interp(val, llat, llon, no_weighting=True)
    per = SNN.interp(per, llat, llon, no_weighting=True)
    dif = SNN.interp(dif, llat, llon, no_weighting=True)
    dis = SNN.interp(dis, llat, llon, no_weighting=True)

    return llon, llat, rad, val, per, dif, dis


if __name__ == "__main__":

    filename = sys.argv[1]

    read_specfem_xsec_depth(filename)
