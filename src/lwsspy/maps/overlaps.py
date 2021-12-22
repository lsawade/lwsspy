
from jax import jit
from re import S

import numpy as np
from .overlap import overlap


def overlaps(segments):
    len_segs = len(segments)
    # Creating counter matrix
    idx = np.zeros((len_segs, len_segs), dtype=bool)

    # Segment overlap

    for _i, x in enumerate(segments):
        for _j, y in enumerate(segments[_i+1:]):
            # print(overlap(x[:2], x[2:], y[:2], y[2:]))

            idx[_i, _i+1+_j] = overlap(x[:2], x[2:], y[:2], y[2:])

    return idx


if __name__ == '__main__':
    pass
