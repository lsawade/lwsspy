import numpy as np


def pixels2data(px, pxi: float, pxf: float, di: float, df: float, log: bool = False):
    """Rescales linearly or with logscale in the data pixel values to data 
    values from a graph by using the image extent and graph extent on one axis. 

    Parameters
    ----------
    px : numpy.ndarray
        Pixel values to be converted
    pxi : float
        pixel starting value
    pxf : float
        pixel end value
    di : float
        data starting limit
    df : float
        data ending limit
    log : bool, optional
        log scale flag, define whether the data axis is in logarithmic scale, 
        by default False

    Returns
    -------
    numpy.ndarray
            data values corresponding to picked data values

    See Also
    --------
    lwsspy.plot.pick_data_from_image.pick_data_from_image : For usage of the function

    Notes
    -----

    .. note::

        For the log scale the idea is fairly simple. Each axis can be viewed
        as a linear semilog plot of pixel-axis vs. data-axis. From there it is
        simple to invert for the scaling by computing slope and intercept.


    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.06 11.00



    """

    # Convert from pixel space to data space
    if log:
        # Find scaling for logarithmic y axis
        m = (np.log10(df) - np.log10(di)) / (pxf - pxi)
        b = np.log10(df) - m * pxf

        # Convert y pixel values
        d = 10 ** (m * px + b)
    else:
        # Find scaling for logarithmic y axis
        m = (df - di) / (pxf - pxi)
        b = di - m * pxi

        # Convert y pixel values
        d = m * px + b

    return d
