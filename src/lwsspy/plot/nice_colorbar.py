import matplotlib
import matplotlib.pyplot as plt
import lwsspy.utils as lutils


def nice_colorbar(*args, fig: bool = False, **kwargs) -> matplotlib.colorbar.Colorbar:
    """Creates nicely formatted colorbar. ``*args`` and ``**kwargs`` are parsed 
    to ``plt.colorbar(*args, **kwargs)``

    Returns
    -------
    matplotlib.colorbar.Colorbar
        colorbar handle returned


    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.01.06 11.00
    """

    # Get normal axes labelsize
    xticklabelsize = matplotlib.rcParams['xtick.labelsize']
    newlabelsize = lutils.reduce_fontsize(xticklabelsize)

    # Change label size to a good size: 70 % of axes label size
    c = plt.colorbar(*args, **kwargs)
    c.ax.tick_params(labelsize=newlabelsize)
    c.ax.yaxis.label.set_size(newlabelsize)
    c.ax.xaxis.label.set_size(newlabelsize)
    c.ax.yaxis.offsetText.set_fontsize(newlabelsize)

    return c
