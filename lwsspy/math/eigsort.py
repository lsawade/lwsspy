import numpy as np


def eigsort(mat: np.ndarray):
    """Returns sorted eigenvalues of a given matrix

    Args:
        mat (np.array): Square matrix

    Returns:
        sorted eigenvalue [vector], and -vectors [matrix]

    Last modified: Lucas Sawade, 2020.09.14 23.00 (lsawade@princeton.edu)
    """

    # Get Eigenvalues
    vals, vecs = np.linalg.eigh(mat)

    # Sort them
    order = vals.argsort()[::-1]

    # Return sorted values and vectors

    return vals[order], vecs[:, order]
