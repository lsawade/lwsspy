from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from .optimizer import Optimization


def plot_optimization(optim: Union[List[Optimization], Optimization],
                      outfile: str or None = None):
    """Plotting Misfit Reduction

    Parameters
    ----------
    optim : Optimization
        Optimization class
    outfile : str or None, optional
        Define where to save the figure. If None, plot is shown,
        by default None
    """
    if type(optim) is not list:
        optim = [optim]
    # Plot values
    plt.figure(figsize=(6, 5))
    ax = plt.axes()
    ax.set_yscale("log")
    for _opt in optim:
        # Get values
        c = _opt.fcost_hist
        it = np.arange(len(c))
        plt.plot(it, c, label=_opt.type.upper())
    plt.legend(loc=1)
    plt.title("Misfit Reduction")

    if outfile is not None:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()
