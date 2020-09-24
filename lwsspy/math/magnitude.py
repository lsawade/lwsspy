from numpy import log10


def magnitude(x: float):
    """Returns the magnitude of a number

    Args:
        x (float):
            just a float

    Returns"
        int:
            Magnitude of x

    """
    return int(log10(x))
