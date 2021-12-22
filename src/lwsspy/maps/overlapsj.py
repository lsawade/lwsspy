
from jax import jit
from re import S

# import numpy as np
# from .overlap import overlap

import jax.numpy as jnp
from jax.lax import fori_loop as foril
from jax.ops import index_update as jidx_update
from jax.ops import index as jidx
from .overlapj import overlapj


def overlapsj(segments):
    len_segs = len(segments)

    # Creating counter matrix
    idx = jnp.zeros((len_segs, len_segs), dtype=jnp.bool_)

    def func_in_i(_i, idx1):
        x = segments[_i, :]

        def func_in_j(_j, idx2):
            y = segments[_j, :]
            idx2 = jidx_update(idx2, jidx[_i, _i+1+_j],
                               overlapj(x[:2], x[2:], y[:2], y[2:]))

            return idx2

        idx1 = foril(_i+1, len_segs, func_in_j, idx1)

        return idx1

    idx = foril(0, len_segs, func_in_i, idx)

    return idx


if __name__ == '__main__':
    pass
