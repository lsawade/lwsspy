import numpy as np
from numba import njit, prange
from numba import jit


def wasserstein(x, f, y, g, p=2):
    # Marginal CDFs
    cf = np.cumsum(f)
    cf /= cf[-1]
    cg = np.cumsum(g)
    cg /= cg[-1]

    # Create tk vector to sample the inverse CDFs
    a = np.append(cf[:-1], cg)
    tkarg = np.argsort(a)
    tk = a[tkarg]

    # Get indeces for each CDF
    indF = np.searchsorted(cf, tk, side='left', sorter=None)
    indG = np.searchsorted(cg, tk, side='left', sorter=None)

    # Get the dt0 vector
    dtk = np.insert(tk[1:] - tk[:-1], 0, tk[0])

    # Compute differnce
    zf = x[indF]
    zg = y[indG]
    delta_z = np.abs(zf-zg)**p

    return np.sum(delta_z*dtk)


@jit(nopython=True, parallel=True)
def compute_dWdp(tk, tkarg, f, iF, dz, p=1):
    """This function computes the gradient of the Wasserstein
    distance wrt. to the 1D source probability density function.
    That is it goes throught the motion of computing B.2 from Sambridge et al. (2022)

    I'm pretty sure that the notation in the paper is wrong for B.4
    The derivative  \partial t_{k-1}/\partial{f_i} is missing. A finite
    difference test (perturbing f_{i}) to compute dW1/df_i, confirmed this.

    """

    # Get all elements that are related to F only
    # See B.4 qk
    qk = (tkarg < f.size-1).astype(np.float64)
    qk[0] = 0

    # Get normalizing factor
    Nf = np.sum(f)

    # normalize f
    fn = f/Nf

    # Initialize
    dWdfi = np.zeros(f.size)

    # Loop over derivative
    for i in prange(f.size):

        # Set dt_{k-1}dfi to zero and update it with the following value
        dtkm1_dfi = 0

        for k in range(1, tk.size):

            # Initialize Derivative of cdf at tk wrt.
            dFik_dfi = 0

            for j in range(f.size):
                # === k
                if iF[k] >= j:
                    dFik_dfj = 1
                else:
                    dFik_dfj = 0

                # B.5 & B.6
                dfj_dfi = (np.float64((i == j)) - fn[j])/Nf

                # B.7
                dFik_dfi += dFik_dfj * dfj_dfi

            # B.4
            dtk_dfi = qk[k] * dFik_dfi

            # B.2
            dWdfi[i] += dz[k]**p * (dtk_dfi-dtkm1_dfi)

            # This term is not missing from B.4 or B.2, it's just absorbed into
            # the matrix A
            dtkm1_dfi = 1*dtk_dfi

    return dWdfi
