import numpy as np


def logistic(
        x, x0: float = 0.0, t: float = 1.0, k: float = 1.0, b: float = 0.0):
    """Customizable logistic function. You can define stretch center top and 
    bottom.

    Parameters
    ----------
    x : float or arraylike
        x values
    x0 : float, optional
        center of the , by default 0.0
    t : float, optional
        topline, by default 1.0
    k : float, optional
        length from baseline to max, by default 1.0
    b : float, optional
        baseline shift, by default 0.0

    Returns
    -------
    like x 
        logistic function output


    Notes
    -----

    A twist on the classic logistic function to be able to set base- and topline
    for simple modification. The function is defined as 

    .. math:: :label: logistic

        f(x)= \\frac{t-b}{1+e^{-\\frac{2 \\pi}{k}\\left(x-x_{0}\\right)}} + b,


    where :math:`b` is the baseline, and :math:`t` is the topline.
    :math:`k` is defined as the length :math:`\\Delta x` from :math:`x_0` to 
    where :math:`f(x) \\sim t` or :math:`f(x) \\sim b`. Hence, I call it a twist
    on the classic logistic function because k is no longer defining the slope.
    The original :math:`k_{\\mathrm{original}}` is defined as

    .. math:: :label: mod

        k_{\\mathrm{original}} = \\frac{2*\\pi}{k}.


    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.10.12 22.00

    """

    return (t-b)/(1 + np.exp(-1/k*2*np.pi*(x-x0))) + b
