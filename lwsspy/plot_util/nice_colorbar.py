import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def nice_colorbar(*args, **kwargs) -> matplotlib.colorbar.Colorbar:
    """Creates nicely formatted colorbar. `**kwargs` are parsed to the 

    Returns
    -------
    matplotlib.colorbar.Colorbar
        colorbar handle returned
    """

    # Get normal axes labelsize
    xticklabelsize = matplotlib.rcParams['xtick.labelsize']
    newlabelsize = int(np.round(0.7*xticklabelsize))

    # Change label size to a good size: 70 % of axes label size
    c = plt.colorbar(*args, **kwargs)
    c.ax.tick_params(labelsize=newlabelsize)
    c.ax.yaxis.label.set_size(newlabelsize)
    c.ax.xaxis.label.set_size(newlabelsize)
    c.ax.yaxis.offsetText.set_fontsize(newlabelsize)

    return c
