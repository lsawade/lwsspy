import numpy as np
from numba import njit, prange
from numba import jit


@jit(nopython=True, parallel=True)
def fingerprint(t: np.ndarray, u: np.ndarray, nt: int, nu: int):

    # Define p grid
    pt = np.linspace(np.min(t), np.max(t), nt)
    pu = np.linspace(0, 1, nu)

    # Get length:
    nk = len(u)-1

    # Initialize d
    d = np.zeros((nt, nu), dtype=np.float64)

    # Initialize arrays
    l = np.zeros(nk, dtype=np.float64)
    x0 = np.zeros(2, dtype=np.float64)
    x1 = np.zeros(2, dtype=np.float64)
    p = np.zeros(2, dtype=np.float64)

    # Loop over grid points and segments
    for i in prange(nt):
        for j in range(nu):

            # Reassign p
            p[0] = pt[i]
            p[1] = pu[j]

            # Zero out l
            l = np.zeros(nk, dtype=np.float64)

            for k in range(nk):
                # Get segment coords
                x0[0] = t[k]
                x0[1] = u[k]
                x1[0] = t[k+1]
                x1[1] = u[k+1]

                # Get lambda
                dx = x1 - x0
                dx2 = dx[0]**2 + dx[1]**2
                lmd = ((p[0] - x0[0])*dx[0] + (p[1] - x0[1])*dx[1]) / dx2

                if lmd > 1:
                    lmd = 1
                elif lmd < 0:
                    lmd = 0

                # Get local dist
                dpx = p - ((1-lmd)*x0 + lmd*x1)
                dpx2 = dpx[0]**2 + dpx[1]**2
                l[k] = np.sqrt(dpx2)

            d[i, j] = np.min(l)

    return pt, pu, d


@jit(nopython=True, parallel=True)
def fingerprint_and_diff(t: np.ndarray, u: np.ndarray, nt: int, nu: int):

    # Define p grid
    pt = np.linspace(np.min(t), np.max(t), nt)
    pu = np.linspace(0, 1, nu)

    # Get length of u and subtract one to get length of segments
    nk = u.size-1

    # Initialize d
    d = np.zeros((nt, nu), dtype=np.float64)
    dij_k = np.zeros((nt, nu), dtype=np.float64)
    dij_kp1 = np.zeros((nt, nu), dtype=np.float64)
    lambdas = np.zeros((nt, nu), dtype=np.float64)
    idk = np.zeros((nt, nu), dtype=np.int64)

    # Make x vector
    x = np.vstack((t, u))
    # print(x.shape)

    # Loop over grid points and segments
    for i in prange(nt):
        # Initialize
        p = np.zeros(2, dtype=np.float64)
        x0 = np.zeros(2, dtype=np.float64)
        x1 = np.zeros(2, dtype=np.float64)

        for j in range(nu):

            # Reassign p

            p[0] = pt[i]
            p[1] = pu[j]

            # Zero out l, lmds
            ldist = np.zeros(nk, dtype=np.float64)
            lmds = np.zeros(nk, dtype=np.float64)
            xclose = np.zeros((2, nk), dtype=np.float64)

            for k in range(nk):

                # Get segment coords
                x0[0] = t[k]
                x0[1] = u[k]
                x1[0] = t[k+1]
                x1[1] = u[k+1]

                # Get lambda
                dx = x1 - x0
                dx2 = dx[0]**2 + dx[1]**2
                lmds[k] = np.dot(p - x0, dx)/dx2

                if lmds[k] > 1:
                    lmds[k] = 1
                elif lmds[k] < 0:
                    lmds[k] = 0

                # Get local distance
                xclose[:, k] = (1-lmds[k])*x0 + lmds[k]*x1
                # dpx = p - xclose[:, k]
                dpx = p - ((1-lmds[k])*x0 + lmds[k]*x1)
                dpx2 = np.dot(dpx, dpx)
                ldist[k] = np.sqrt(dpx2)

            # C.5
            kl = np.argmin(ldist)
            d[i, j] = ldist[kl]
            lambdas[i, j] = lmds[kl]
            idk[i, j] = kl

            # Compute dddx
            dddx = (xclose[:, kl] - p) / d[i, j]

            # Compute dxduk and dxdukp1
            dxduk, dxdukp1 = compute_dxduk(x[:, kl], x[:, kl+1], p, lmds[kl])

            # Compute ddduk dddukp1
            ddduk = np.dot(dddx, dxduk)
            dddukp1 = np.dot(dddx, dxdukp1)

            # C.9 Error in this equation it should be u(lambda)
            # dij_k[i, j] = (1-lmds[kl])/d[i, j] * (uclose - pu[j])
            # dij_k[i, j] = (1-lmds[kl])/d[i, j] * (u[kl] - pu[j])
            dij_k[i, j] = ddduk

            # C.10 Error in this equation it should be u(lambda)
            # dij_kp1[i, j] = lmds[kl]/d[i, j] * (uclose - pu[j])
            # dij_kp1[i, j] = lmds[kl]/d[i, j]s * (u[kl+1] - pu[j])
            dij_kp1[i, j] = dddukp1

    return pt, pu, d, lambdas, dij_k, dij_kp1, idk


@jit(nopython=True)
def compute_dxduk(xk: np.ndarray, xkp1: np.ndarray, p: np.ndarray, lmd: float):
    """
    Outputs tuple with dxduk and dxdukp1.

    xk   = [tk,   uk  ].T
    xkp1 = [tkp1, ukp1].T
    p    = [tp,   up].T

    x   =  [t,    uk  ].T

    Definition of the segment
        x(lambda, xk, xk+1) = (1-lambda) * xk + lamdba * xkp1

    Some definitions
        lamda = g()/h(),            
    where
        g = (p - xk) dot (xk+1 - xk)
        h = || xk+1 - xk ||^2.

    so dlam/duk = dgduk * h - g * dhduk
    """

    # Get difference
    dx = xkp1 - xk

    # Get numerator and denominator
    g = np.dot(p-xk, dx)
    h = np.dot(dx, dx)

    # Derivatives of the numerator and denominator as a function of uk
    dgduk = - ((xkp1[1] - xk[1]) + (p[1] - xk[1]))
    dhduk = 2 * (xk[1] - xkp1[1])

    # Derivatives of the numerator and denominator as a function of uk
    dgdukp1 = (p[1] - xk[1])
    dhdukp1 = 2 * (xkp1[1] - xk[1])

    # Get derivatives of lambda wrt. uk and ukp1
    dlmdduk = (dgduk * h - g * dhduk)/h**2
    dlmddukp1 = (dgdukp1 * h - g * dhdukp1)/h**2

    # Get derivatives of x wrt. lamda uk and ukp1. Here it's important to note
    # x is directly dependent on uk and ukp1, but also on lambda which itself is
    # dependent on uk
    dxdlmd = dx
    single_dxduk = (1-lmd) * np.array([0, 1])
    single_dxdukp1 = (lmd) * np.array([0, 1])

    # Combining the derivatives
    dxduk = dxdlmd * dlmdduk + single_dxduk
    dxdukp1 = dxdlmd * dlmddukp1 + single_dxdukp1

    return dxduk, dxdukp1
