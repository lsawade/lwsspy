import numpy as np


def get_stats_label_length(mu: float, sig: float, ndec: int = 2):
    """Gets format specifier length to print symmetric labels into the 
    figure. 

    Parameters
    ----------
    mu : float
        mean
    sig : float
        standard deviation
    ndec : int, optional
        number of decimals, by default 2

    Returns
    -------
    tuple(int, int)
        (total length, number of decimals)
    """

    # Get mu length
    muint = np.floor(np.log10(np.abs(mu)))
    if muint <= 0:
        muint = 1
    if mu < 0:
        muint += 1

    # Get sigma lenf
    sigint = np.floor(np.log10(sig))
    if sigint <= 0:
        sigint = 1

    # create combined and decimal
    if ndec == 0:
        extra = 0
    else:
        extra = 1 + ndec

    # Create combined label
    labint = int(np.maximum(sigint, muint) + extra)

    return labint, ndec
