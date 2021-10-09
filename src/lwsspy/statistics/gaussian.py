import numpy as np


def gaussian(t, to, sig, amp):
    """Create Gaussian pulse.

    Parameters
    ----------

    t
        time vector
    to 
        time in the center
    sig 
        standard deviation
    amp 
        amplitude

    Returns
    -------
    arraylike
        vector with same size of t containing the corresponding pulse


    Notes
    -----

    .. math:: :label: gaussian

        f(a, x, t_0, \sigma) = ae^{-0.5 \\left[\\frac{t-t_0}{\sigma}\\right]^2}

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.10.12 22.00

    """

    return amp * np.exp(-0.5 * (t-to)**2 / (sig * sig))
