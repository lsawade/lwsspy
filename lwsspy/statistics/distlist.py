from typing import List, Tuple, Union
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def distlist(N: Union[None, int] = None,
             plot: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Creates list of some distributions

    Args:
        N (Union[None, int], optional):
            Number of convolutions to be done. Defaults to None.
        plot (bool, optional):
            Plotting flag. Defaults to True.

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: x vector, list of distributions

    Last modified: Lucas Sawade, 2020.09.30 11.00 (lsawade@princeton.edu)
    """

    # Location of the distributions
    x = np.linspace(-5, 5, 15)

    # Empty list of distributions
    dists = []

    # # Create Normal distribution
    # locs = [-4, 3, 5, 10]
    # scales = [5, 2, 6, 20]

    # for loc, scale in zip(locs, scales):
    #     n = stats.norm(loc=loc, scale=scale)
    #     dists.append(n.pdf(x))

    # Create uniform distrubtions
    locs = [-5, -4, -3, -2, -1]
    scales = [np.abs(x)*2 for x in locs]

    for loc, scale in zip(locs, scales):
        u = stats.uniform(loc=loc, scale=scale)
        dists.append(u.pdf(x))

    # Create random distributions
    for _i in range(5):
        d = np.random.choice(10, size=len(x), replace=True)
        d = d/np.trapz(d, x=x)
        dists.append(d)

    # Shuffle the distrubtions
    dists = [dists[_i] for _i in np.random.choice(
                 len(dists), size=len(dists), replace=False)]
    if N is not None:
        dists = dists[:N]

    if plot:
        plt.figure()
        for _i, dist in enumerate(dists):
            plt.plot(x, dist, label=f"{_i + 1}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"{len(dists)} distributions")
        plt.legend()
        plt.xlim(np.min(x), np.max(x))

    return x, dists


if __name__ == "__main__":

    import os
    import matplotlib.pyplot as plt
    from ..plot_util.updaterc import updaterc
    from .. import DOCFIGURES
    updaterc()

    # Testrun
    x, distr = distlist(N=5, plot=True)
    plt.savefig(os.path.join(DOCFIGURES, 'distlist.svg'))
    plt.show()
