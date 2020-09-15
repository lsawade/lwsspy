import matplotlib


def updaterc(rebuild=False):
    """Updates the rcParams to something generic that looks ok good out of 
    the box.

    Args:

        rebuild (bool):
            Rebuilds fontcache incase it needs it.

    Last modified: Lucas Sawade, 2020.09.15 01.00 (lsawade@princeton.edu)

    """

    params = {
        'font.family': 'Helvetica Neue',
        'pdf.fonttype': 42,
        'font.weight': 'normal',
        'axes.labelweight': 'normal',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 1,
        'axes.grid': True,
        'grid.color': "lightgrey",
        'xtick.labelsize': 10,
        'xtick.direction': 'inout',
        'xtick.top': True,  # draw label on the top
        'xtick.bottom': True,  # draw label on the bottom
        'xtick.minor.visible': True,
        'xtick.major.top': True,  # draw x axis top major ticks
        'xtick.major.bottom': True,  # draw x axis bottom major ticks
        'xtick.major.size': 8,  # draw x axis top major ticks
        'xtick.minor.top': True,  # draw x axis top minor ticks
        'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'xtick.minor.size': 4,  # draw x axis top major ticks
        'ytick.labelsize': 10,
        'ytick.direction': 'inout',
        'ytick.left': True,  # draw label on the top
        'ytick.right': True,  # draw label on the bottom
        'ytick.minor.visible': True,
        'ytick.major.left': True,  # draw x axis top major ticks
        'ytick.major.right': True,  # draw x axis bottom major ticks
        'ytick.major.size': 8,  # draw x axis top major ticks
        'ytick.minor.left': True,  # draw x axis top minor ticks
        'ytick.minor.right': True,  # draw x axis bottom minor ticks
        'ytick.minor.size': 4,  # draw x axis top major ticks
        'legend.fancybox': False,
        'legend.frameon': True,
        'legend.loc': 'best',
        'legend.numpoints': 2,
        'legend.fontsize': 10,
        'legend.framealpha': 1,
        'legend.scatterpoints': 3,
        'legend.edgecolor': 'inherit',
        'legend.facecolor': 'w',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Helvetica',
        'mathtext.it': 'Helvetica:italic',
        'mathtext.bf': 'Helvetica:bold'
    }
    matplotlib.rcParams.update(params)

    if rebuild:
        matplotlib.font_manager._rebuild()
