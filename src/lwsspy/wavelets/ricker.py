import numpy as np


def ricker(t, tc: float = 0.0, f0: float = 1.0, A: float = 1.0):
    """Returns a ricker wavelet with specified peak frequency, timeshift, 
    and amplitude.

    Parameters
    ----------
    t : arraylike
        time vector
    f0 : float, optional
        peak frequency, by default 1.0
    tc : float, optional
        timeshift, by default 0.0
    A : float, optional
        Amplitude, by default 1.0

    Returns
    -------
    r
        array of same size as input t
    """
    return A*(1-2*(np.pi*f0*(t-tc))**2) * np.exp(-(np.pi*f0*(t-tc))**2)


def dr_dt0(t, tc: float = 0.0, f0: float = 1.0, A: float = 1.0):
    """Returns the derivative of a ricker wavelet with specified peak frequency, 
    timeshift, and amplitude wrt. the timeshift tc

    Parameters
    ----------
    t : arraylike
        time vector
    f0 : float, optional
        peak frequency, by default 1.0
    tc : float, optional
        timeshift, by default 0.0
    A : float, optional
        Amplitude, by default 1.0

    Returns
    -------
    r
        array of same size as input t
    """
    return (
        + 4*A * np.pi**2 * f0**2 * (t-tc) * np.exp(-(np.pi*f0*(t-tc))**2)
        + ricker(t, tc, f0, A) * (np.pi**2 * f0**2 * 2*(t-tc))
    )


def dr_df0(t, tc: float = 0.0, f0: float = 1.0, A: float = 1.0):
    """Returns the derivative of a ricker wavelet with specified peak frequency, 
    timeshift, and amplitude wrt. the frequency f0

    Parameters
    ----------
    t : arraylike
        time vector
    f0 : float, optional
        peak frequency, by default 1.0
    tc : float, optional
        timeshift, by default 0.0
    A : float, optional
        Amplitude, by default 1.0

    Returns
    -------
    r
        array of same size as input t
    """
    return (
        - 4*A * np.pi**2 * f0 * (t-tc)**2 * np.exp(-(np.pi*f0*(t-tc))**2)
        - ricker(t, tc, f0, A) * (np.pi**2 * 2*f0 * (t-tc)**2)
    )


def dr_dA(t, tc: float = 0.0, f0: float = 1.0, A: float = 1.0):
    """Returns the derivative of a ricker wavelet with specified peak frequency, 
    timeshift, and amplitude wrt. the frequency f0

    Parameters
    ----------
    t : arraylike
        time vector
    f0 : float, optional
        peak frequency, by default 1.0
    tc : float, optional
        timeshift, by default 0.0
    A : float, optional
        Amplitude, by default 1.0

    Returns
    -------
    r
        array of same size as input t
    """
    return ricker(t, tc, f0, A)/A


def R(t, t0: float = 0.0, f0: float = 1.0, A: float = 1.0, L: float = 2.0):
    """Returns a double ricker wavelet with specified peak frequency, 
    timeshift, and amplitude. This Ricker is taken from Sambridge (2022).
    t0 here is the center between the 2 wavelets.

    Parameters
    ----------
    t : arraylike
        time vector
    L : float, optional
        distance between the rickers, by default 2.0
    t0 : float, optional
        timeshift, by default 0.0
    f0 : float, optional
        peak frequency, by default 1.0
    A : float, optional
        Amplitude, by default 1.0

    Returns
    -------
    R
        array of same size as input t
    """

    # Get t1 and t0
    t1 = (t0 - L/2)
    t2 = (t0 + L/2)

    return ricker(t, t1, f0, A) + ricker(t, t2, f0, A)


def dRdm(t, t0: float = 0.0, f0: float = 1.0, A: float = 1.0, L: float = 2.0):
    """Returns the gradient of a double ricker wavelet with specified
    peak frequency, timeshift, and amplitude with respect to the
    model parameters (t0, f0, A) in this order. This Ricker is taken from
    Sambridge (2022). t0 here is the center between the 2 wavelets.


    Parameters
    ----------
    t : arraylike
        time vector
    L : float, optional
        distance between the rickers, by default 2.0
    t0 : float, optional
        timeshift of the double ricker, by default 0.0
    f0 : float, optional
        peak frequency, by default 1.0
    A : float, optional
        Amplitude, by default 1.0

    Returns
    -------
    tuple 
        dRdm
    """

    # Get t1 and t0
    t1 = (t0 - L/2)
    t2 = (t0 + L/2)

    # 'Needed' chain rule artial derivatives
    dt1_dt0 = 1.0
    dt2_dt0 = 1.0
    dt1_dL = -0.5
    dt2_dL = +0.5

    return (
        dr_dt0(t, t1, f0, A) * dt1_dt0 + dr_dt0(t, t2, f0, A) * dt2_dt0,
        dr_df0(t, t1, f0, A) + dr_df0(t, t2, f0, A),
        dr_dA(t, t1, f0, A) + dr_dA(t, t2, f0, A),
        dr_dt0(t, t1, f0, A) * dt1_dL + dr_dt0(t, t2, f0, A) * dt2_dL,
    )
