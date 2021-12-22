import numpy as np
import matplotlib.pyplot as plt


def rectangle_from_extent(extent, *args, **kwargs):

    # Create array for simple processing
    extent = np.array(extent)

    # Compute widht and height
    width, height = np.diff(extent[:2]), np.diff(extent[2:])
    patch = plt.Rectangle(extent[0::2], *args,
                          width=width, height=height, **kwargs)

    return patch
