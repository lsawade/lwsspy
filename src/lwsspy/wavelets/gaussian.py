import numpy as np
from scipy.integrate import cumtrapz


def G(t, t0=0.0, f0=20.0, dt=1.0):
    """Integrated Gaussian curve with dominant frequency and time-shift

    Parameters
    ----------
    t : Array-like
        time vector
    t0 : float, optional
        timeshift, by default 0.0
    f0 : float, optional
        dominant frequenncy in Hz, by default 20.0

    Returns
    -------
    Arraylike
        output gaussian

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.03.02 23.30

    """
    return cumtrapz(np.exp(-(4*f0)**2 * (t-t0)**2), t)


def g(t, t0=0.0, f0=20.0):
    """Gaussian curve with dominant frequency and time-shift

    Parameters
    ----------
    t : Array-like
        time vector
    t0 : float, optional
        timeshift, by default 0.0
    f0 : float, optional
        dominant frequenncy in Hz, by default 20.0

    Returns
    -------
    Arraylike
        output gaussian

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.03.02 23.30

    """
    return np.exp(-(4*f0)**2 * (t-t0)**2)


def dgdt(t, t0=0.0, f0=20.0):
    """Derivative of gaussian curve with dominant frequency and time-shift

    Parameters
    ----------
    t : Array-like
        time vector
    t0 : float, optional
        timeshift, by default 0.0
    f0 : float, optional
        dominant frequenncy in Hz, by default 20.0

    Returns
    -------
    Arraylike
        output gaussian

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.03.02 23.30

    """

    return -(8*f0) * (t-t0) * np.exp(-(4*f0)**2 * (t-t0)**2)
