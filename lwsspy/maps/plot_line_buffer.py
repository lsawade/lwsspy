from .line_buffer import line_buffer
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


def plot_line_buffer(lat, lon, delta: float = 1, **kwargs):
    """Takes in 

    Parameters
    ----------
    lat : np.ndarray
        latitudes of a line
    lon : np.ndarray
        longitudes of a line
    delta : float, optional
        epicentral distance of the buffer, by default 1

    Returns
    -------
    tuple
        (patch, artist)
    """

    # Get axes
    ax = plt.gca()

    # Get buffer
    poly, circles = line_buffer(lat, lon, delta=delta)

    # Plot into figure
    mpolys = []
    artists = []
    for _poly in poly:
        mpoly = Polygon(_poly, **kwargs)
        mpolys.append(mpoly)
        artists.append(ax.add_patch(mpoly))

    return poly, circles, mpolys, artists
