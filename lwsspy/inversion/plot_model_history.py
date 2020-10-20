
import numpy as np
import matplotlib.pyplot as plt
from .optimizer import Optimization


def plot_model_history(optim: Optimization, labellist: list or None = None,
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

    # Get model parameter number
    ncol = int(np.ceil(optim.n))
    mrow = int(np.ceil(optim.n/ncol))

    # Get values
    c = optim.fcost_hist
    it = np.arange(len(c))

    # Plot values
    plt.figure(figsize=(5*ncol/mrow, 5))

    for _i in range(optim.n):

        ax = plt.subplot(mrow, ncol, _i + 1)

        if labellist is not None:
            label = labellist[_i]
        else:
            label = f"Model Param: {_i:{len(str(optim.n))}}"

        plt.plot(np.arange(optim.current_iter + 1),
                 optim.msave[_i, :optim.current_iter + 1],
                 label=label)
        plt.legend()
        plt.title(f"{optim.type.upper()} - Misfit Reduction")

        if outfile is not None:
            plt.savefig(outfile, dpi=300)
        else:
            plt.show()
