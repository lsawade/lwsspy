import numpy as np


def in_extent(xmin, xmax, ymin, ymax, x, y):

    # Check wether array in extent
    pos = np.where((xmin <= x) & (x <= xmax) &
                   (ymin <= y) & (y <= ymax))[0]

    if len(pos) > 0:
        return True
    else:
        return False
