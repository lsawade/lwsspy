import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def topocolormap():
    """Returns cut version of matplotlibs terrain colormap.

    Taken from: https://stackoverflow.com/questions/40895021/python-equivalent-for-matlabs-demcmap-elevation-appropriate-colormap

    Returns
    -------
    matplotlib.colormap
        colormap
    """

    # Combine the lower and upper range of the terrain colormap with a gap in the middle
    # to let the coastline appear more prominently.
    # inspired by https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps
    colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 224))
    colors_land1 = plt.cm.terrain(np.linspace(0.25, 0.5, 20))
    colors_land2 = plt.cm.terrain(np.linspace(0.5, 0.75, 200))
    colors_land3 = plt.cm.terrain(np.linspace(0.75, 0.825, 340))
    colors_land4 = plt.cm.terrain(np.linspace(0.825, 1.0, 240))
    # combine them and build a new colormap
    colors = np.vstack((colors_undersea, colors_land1,
                        colors_land2, colors_land3, colors_land4))
    cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        'cut_terrain', colors)

    return cut_terrain_map
