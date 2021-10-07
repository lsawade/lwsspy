import numpy as np


def power_l1(arr1, arr2):
    """
    Power(L1 norm, abs) ratio of arr1 over arr2, unit in dB
    """
    if len(arr1) != len(arr2):
        raise ValueError("Length of arr1(%d) and arr2(%d) not the same"
                         % (len(arr1), len(arr2)))
    return 10 * np.log10(np.sum(np.abs(arr1)) / np.sum(np.abs(arr2)))


def power_l2(arr1, arr2):
    """
    Power(L2 norm, square) ratio of arr1 over arr2, unit in dB.
    """
    if len(arr1) != len(arr2):
        raise ValueError("Length of arr1(%d) and arr2(%d) not the same"
                         % (len(arr1), len(arr2)))
    return 10 * np.log10(np.sum(arr1 ** 2) / np.sum(arr2 ** 2))
