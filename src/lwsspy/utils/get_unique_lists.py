from typing import Tuple
import numpy as np


def get_unique_lists(*args: list) -> Tuple[list]:
    """Takes in individual vectors (lists) of the same length and checks them 
    for combined uniques.

    Parameters
    ----------
    *args : list
        lists in form of individual inputs, say x, y, z coordinates.


    Returns
    -------
    Tuple[list]
        [description]

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last modifided:
        Lucas Sawade 2021.01.14 13.00
    """

    # Create array from lists
    array = np.array([*args]).T

    # Get unique rows
    unique_rows = np.unique(array, axis=0)

    out_list = []
    for col in unique_rows.T:
        out_list.append(col.tolist())

    return tuple(out_list)
