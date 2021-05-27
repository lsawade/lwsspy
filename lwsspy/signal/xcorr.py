import numpy as np


def xcorr(d, s):
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    # Normalized cross correlation.
    max_cc_value = cc.max() / np.sqrt((s ** 2).sum() * (d ** 2).sum())
    return max_cc_value, time_shift


def correct_window_index(istart, iend, nshift, npts):
    """Correct the window index based on cross-correlation shift

    Parameters
    ----------
    istart : int
        start index
    iend : int
        end index
    nshift : int
        shift in N samples
    npts : int
        Length of window

    Returns
    -------
    Tuple
        indeces

    Raises
    ------
    ValueError
        If resulting windows arent the same length? I don't get this
    """
    istart_d = max(1, istart + nshift)
    iend_d = min(npts, iend + nshift)
    istart_s = max(1, istart_d - nshift)
    iend_s = min(npts, iend_d - nshift)
    if (iend_d - istart_d) != (iend_s - istart_s):
        raise ValueError("After correction, window length not the same: "
                         "[%d, %d] and [%d, %d]" % (istart_d, iend_d,
                                                    istart_s, iend_s))
    return istart_d, iend_d, istart_s, iend_s
