from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from .optimizer import Optimization


def plot_model_history(optim: Union[List[Optimization], Optimization],
                       labellist: Union[list, None] = None,
                       outfile: Union[str, None] = None):
    """Plotting Misfit Reduction

    Parameters
    ----------
    optim : Optimization
        Optimization class
    outfile : str or None, optional
        Define where to save the figure. If None, plot is shown,
        by default None
    """

    # Change type to list
    if isinstance(optim, Optimization):
        optim = [optim]
    elif isinstance(optim, list):
        pass
    else:
        raise ValueError("Wrong")

    # Get model parameter number
    ncol = int(np.ceil(optim[0].n))
    mrow = int(np.ceil(optim[0].n/ncol))

    # Plot values
    plt.figure(figsize=((6*ncol-1, 6*mrow-1)))

    for _i in range(optim[0].n):

        ax = plt.subplot(mrow, ncol, _i + 1)

        if labellist is not None:
            title = labellist[_i]
        else:
            title = f"Model Param: {_i:{len(str(optim[0].n))}}"

        for _opt in optim:
            plt.plot(np.arange(_opt.current_iter + 1),
                     _opt.msave[_i, :_opt.current_iter + 1],
                     label=_opt.type.upper())
        plt.title(title)
        plt.legend()

        if outfile is not None:
            plt.savefig(outfile, dpi=300)
        else:
            plt.show()
