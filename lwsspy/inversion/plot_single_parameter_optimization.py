from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from .optimizer import Optimization
import lwsspy as lpy


def plot_single_parameter_optimization(
        optim: Union[List[Optimization], Optimization],
        modellabel: str = "Model", outfile: str or None = None):
    """Plotting Misfit Reduction and Model history.

    Parameters
    ----------
    optim : Optimization
        Optimization class
    outfile : str or None, optional
        Define where to save the figure. If None, plot is shown,
        by default None

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.03.07 01.30

    """
    if type(optim) is not list:
        optim = [optim]
    # Plot values
    plt.figure(figsize=(8, 4))
    plt.subplots_adjust(wspace=0.35)
    ax = plt.subplot(121)
    lpy.plot_label(ax, "a)", location=6, box=False)
    for _opt in optim:
        # Get values
        c = _opt.fcost_hist
        it = np.arange(len(c))
        plt.plot(it, c, label=_opt.type.upper())
    ax.set_yscale('log')
    plt.legend(frameon=False, loc='upper right')
    plt.xlabel("Iteration N")
    plt.ylabel("Norm. Misfit")

    ax1 = plt.subplot(122, sharex=ax)
    lpy.plot_label(ax1, "b)", location=6, box=False)
    for _opt in optim:
        plt.plot(np.arange(_opt.current_iter + 1),
                 _opt.msave[0, :_opt.current_iter + 1],
                 label=_opt.type.upper())
    plt.legend(frameon=False, loc='upper right')
    plt.xlabel("Iteration N")
    plt.ylabel(modellabel)

    if outfile is not None:
        plt.savefig(outfile)
    else:
        plt.show()
