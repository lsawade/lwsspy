"""Set of functions that remove ticks and labels in axes locations

:Author:
    Lucas Sawade (lsawade@princeton.edu)

:Last Modified:
    2020.01.13 14.30

"""

from matplotlib.axes import Axes


def remove_xticklabels(ax: Axes):
    """Removes xticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(labelbottom=False, labeltop=False)


def remove_yticklabels(ax: Axes):
    """Removes yticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(labelleft=False, labelright=False)


def remove_ticklabels(ax: Axes):
    """Removes xticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(labelbottom=False, labeltop=False,
                   labelleft=False, labelright=False)


def remove_ticks(ax: Axes):
    """Removes xticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(bottom=False, top=False,
                   left=False, right=False)

def remove_ticklabels_topright(ax: Axes):
    """Removes top and right ticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(labeltop=False, labelright=False)


def remove_ticklabels_bottomleft(ax: Axes):
    """Removes bottom and left ticklabels of an axes

    Args:
        ax (Axes): Axes handles
    """
    ax.tick_params(labelbottom=False,
                   labelleft=False)
