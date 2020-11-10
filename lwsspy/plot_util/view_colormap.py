"""
Conveniently take from:

https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html

Last modified: Lucas Sawade, 2020.11.04 09.30 (lsawade@princeton.edu)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def grayscale_cmap(cmap):
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)


def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(yticks=[]))
    ax[0].imshow([colors], extent=[0, 1, 0, 1], aspect=0.1)
    ax[0].tick_params(labelbottom=False)
    ax[1].imshow([grayscale], extent=[0, 1, 0, 1], aspect=0.1)
