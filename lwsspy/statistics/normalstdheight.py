from scipy.stats import norm


def normalstdheight(e: float = 0.0, s: float = 1.1, N: float = 1.0) -> float:
    """Gets the height of the normal distrubtions at given number of standard
    deviations.

    Args:
        e (float, optional): [description]. Defaults to 0.0.
        s (float, optional): [description]. Defaults to 1.1.
        N (float, optional): [description]. Defaults to 1.0.

    Returns:
        float: height of the Gaussian 

    Last modified: Lucas Sawade, 2020.09.30 11.00 (lsawade@princeton.edu)
    """
    return norm(loc=e, scale=s).pdf(N*s+e)

