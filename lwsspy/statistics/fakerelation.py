import os
from typing import List
import numpy as np  # type: ignore


def fakerelation(C12: List[float] = [1.0, 0.5, 1.0],
                 N: int = 25,
                 distro: str = 'normal',
                 pars: dict = {'loc': 0.0, 'scale': 1.0}):
    """
    Creates data pairs with a particular relation.

    Args:
        C12 (list, optional):
            Covariance vector [CXX, CXY, CYY]. Defaults to [1, 0.5, 1].
        N (int, optional):
            Number of values. Defaults to 100.
        distro (str, optional):
            Name of desired distribution. Defaults to 'norm'.
        pars (dict, optional):
            Distribution Parameters must match distribution input parameter.
            Check `numpy.random.<dist>`

    Returns:
        (X,Y, R, CXY) (numpy.ndarray, numpy.ndarray,
                       numpy.ndarray, numpy.ndarray):
            Two variables with correlation and their correlation and covariance
            matrices.

    Last modified: Lucas Sawade, 2020.09.14 23.00 (lsawade@princeton.edu)

    """

    # Define the population covariance matrix
    C12 = np.array([[C12[0], C12[1]], [C12[1], C12[2]]])

    # Calculate the initial uncorrelated random variables
    distfunc = getattr(np.random, distro)
    Z1 = distfunc(**pars, size=N)
    Z2 = distfunc(**pars, size=N)

    # Calculate the transformation matrix
    L = np.linalg.cholesky(C12)

    # Calculate the joint pair
    XY = (L.T @ np.vstack((Z1, Z2))).T

    # Distribute over the variables
    X = XY[:, 0]
    Y = XY[:, 1]

    # Report on their sample covariance
    CXY = np.cov(X, Y)

    # Report on their Correlation
    R = np.corrcoef(X, Y)

    return X, Y, R, CXY


if __name__ == "__main__":

    # Testrun 
    X, Y, R, CXY = fakerelation(C12=[1.0, 0.5, 1.0], N=100, distro='normal',
                                pars={'loc': 0.0, 'scale': 1.0})

    import matplotlib.pyplot as plt  # type: ignore
    from ..plot_util.updaterc import updaterc  # type: ignore
    from .. import DOCFIGURES  # type: ignore
    updaterc()


    # Define where the text for correlation coefficient should go
    textdict = {"horizontalalignment": "left",
                "verticalalignment": "top"}

    plt.figure()
    plt.plot(X, Y, 'o')
    plt.text(0.025, 0.975, f"R = {R[0,1]}", transform=plt.gca().transAxes,
             **textdict)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelled covarying data set.')
    plt.savefig(os.path.join(DOCFIGURES, 'fakerelation.svg'))
    plt.show()
