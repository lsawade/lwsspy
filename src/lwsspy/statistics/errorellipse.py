import os
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Local imports
from ..math.eigsort import eigsort
from ..statistics.fakerelation import fakerelation


def errorellipse(x, y, nstd: int = 2,
                 ax: Union[None, plt.Axes] = None,
                 **kwargs):
    """Plots error ellipse

    Args:
        x (np.ndarray or list):
            N element list
        y (np.ndarray or list):
            N element list
        kwargs:
            parsed to matplotlibs Ellipse function

    Returns:
        ellipse handle

    Last modified: Lucas Sawade, 2020.09.15 15.30 (lsawade@princeton.edu)

    """

    # Get covariance and eigenvectors
    cov = np.cov(x, y)
    vals, vecs = eigsort(cov)

    # Get angle from the vectors
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Get width and height of the ellipse
    w, h = 2 * nstd * np.sqrt(vals)

    # Create elliptical patch
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=w, height=h,
                  angle=theta, **kwargs)

    # Get axes
    if ax is None:
        ax = plt.gca()

    # Add patch
    ax.add_patch(ell)

    # return ellipse's handle
    return ell


if __name__ == "__main__":

    X, Y, R, CXY = fakerelation(C12=[1.0, 0.5, 1.0], N=100, distro='normal',
                                pars={'loc': 0.0, 'scale': 1.0})

    import matplotlib.pyplot as plt
    from ..plot.updaterc import updaterc
    from .. import DOCFIGURES
    updaterc()

    # Define where the text for correlation coefficient should go
    textdict = {"horizontalalignment": "left",
                "verticalalignment": "top",
                "zorder": 100}

    plt.figure()
    ax = plt.gca()
    plt.plot(X, Y, 'ko', zorder=10, label="Data")
    plt.text(0.025, 0.975, f"R = {R[0,1]}", transform=plt.gca().transAxes,
             **textdict)
    errorellipse(X, Y, nstd=1, ax=ax, color="grey", zorder=5, alpha=0.5,
                 label=r"$1\sigma$")
    errorellipse(X, Y, nstd=2, ax=ax, color="lightgrey", zorder=2, alpha=0.5,
                 label=r"$2\sigma$")
    plt.legend(loc="lower right", fancybox=False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelled covarying data set.')
    plt.savefig(os.path.join(DOCFIGURES, 'error_ellipse.svg'))
    plt.show()
