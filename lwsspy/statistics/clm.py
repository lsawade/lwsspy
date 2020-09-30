# External 
from typing import List, Tuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Internal
from .normalstdheight import normalstdheight
from ..math.convm import convm


def clm(x: np.ndarray, distr: List[np.ndarray],
        N: int = 1, plot: bool = True) -> Tuple[float, float, float]:
    """Function that takes a list of distributions of length N and shows that
    the central limit theorem holds.

    Args:
        x (np.ndarray):
            x location of the distributions
        distr (List[np.ndarray]):
            List of discrete PDFs
        N (int, optional):
            Number of convolutions to be performed by iteration over
            the list of distributions. Defaults to 1.
        plot (bool, optional):
            plotting flag. Defaults to True.

    Raises:
        ValueError: Raised of list of distributions only contains one
                    distribution.

    Returns:
        Tuple[float, float, float]: 
            Expected Value, Variance, cost sum of squares compared to Normal D.
    
    **Note**:
        The model plot below is not technically correct, as marked in the
        code, it is just for illustration of the shape. The correct
        implementation of the convolution operation is commented.

    Last modified: Lucas Sawade, 2020.09.30 11.00 (lsawade@princeton.edu)

    """


    # Get spacing
    dx = x[1] - x[0]

    if len(distr) < 2:
        raise ValueError("You need at least 2 distributions")
    Ndist = len(distr)

    # Setup first convolution
    Ndist = len(distr)

    # Collect values and misfits
    epdf_list = []
    vpdf_list = []
    phi_list = []

    # Counter used to control the index and number of convolutions
    _i = 0

    # 
    for _j in range(N):
        # for _i in range(1, Ndist):
        # Get new values
        if _j == 0:
            ppdf = distr[_i]
        ppdf1 = distr[_i + 1]

        ppdf = np.convolve(ppdf, ppdf1, mode='full')

        # Theoretically correct implemenation:
        # np.arange(len(ppdf)) * dx + np.min(x)
        # Illustrative implementation:
        x = np.linspace(np.min(x), np.max(x), len(ppdf))
        ppdf = ppdf/np.trapz(ppdf, x=x)

        # Get expected value of PDF
        epdf = np.trapz(x * ppdf, x=x)
        vpdf = np.trapz((x - epdf) ** 2 * ppdf, x=x)

        # Get misfit for each iteration
        forward = stats.norm(loc=epdf, scale=np.sqrt(vpdf)).pdf(x)
        phi = np.sum((forward - ppdf) ** 2)

        # Collect values and misfits
        epdf_list.append(epdf)
        vpdf_list.append(vpdf)
        phi_list.append(phi)

        _i += 1
        if _i == Ndist-1:
            # -1 becuase we want the second and every subsequent 
            # round to start at 0 for ppdf1
            _i = -1

    if plot:

        ax = plt.gca()
        ax.plot(x, ppdf, label="Dist", c='k')
        ax.plot(x, forward, c='red', label="Normal")
        ax.plot([-np.sqrt(vpdf)+epdf, np.sqrt(vpdf)+epdf],
                np.ones(2)*normalstdheight(epdf, np.sqrt(vpdf), N=1), c='b')
        ax.plot([epdf, epdf], [0, np.max(forward)], c='b')
        plt.xlim(np.min(x), np.max(x))
        textdict = {"horizontalalignment": "right", "verticalalignment": "top"}
        plt.text(
            0.95, 0.95,
            f"$\\mu = {epdf:2.2f}$\n$\\sigma = {np.sqrt(vpdf):2.2f}$",
            transform=plt.gca().transAxes, **textdict)
    return epdf, vpdf, phi


if __name__ == "__main__":

    import os
    from ..plot_util.updaterc import updaterc
    from .. import DOCFIGURES
    from .distlist import distlist
    updaterc()

    # Testrun
    x, distr = distlist(N=5, plot=False)
    plt.figure(figsize=(8, 6))
    for _i, ind in enumerate([1, 2, 4, 8]):
        if _i == 0:
            ax = plt.subplot(221+_i)
        else:
            ax = plt.subplot(221+_i, sharex=ax, sharey=ax)
        epdf, vpdf, phi = clm(x, distr, N=ind)
        plt.title(f"Convolutions: {ind}")
        if _i in [0, 2]:
            plt.ylabel("y")
        if _i in [0, 1]:
            ax.tick_params(labelbottom=False)
            # remove_xticklabels(ax)
        if _i in [2, 3]:
            plt.xlabel("x")
        if _i in [1, 3]:
            ax.tick_params(labelleft=False)

    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.savefig(os.path.join(DOCFIGURES, 'clm.svg'))
    plt.show()
