import numpy as np


def rosenbrock(r):
    """Rosenbrock values"""
    x = r[0]
    y = r[1]
    f = (1-x) ** 2 + 100*(y-x*x) ** 2
    return f


def drosenbrock(r):
    """Rosenbrock gradient"""
    x = r[0]
    y = r[1]
    g = np.zeros(2)
    g[0] = -2*(1-x) - 400*x*(y-x*x)
    g[1] = 200*(y-x*x)
    return g


def ddrosenbrock(r):
    """Rosenbrock gradient"""
    x = r[0]
    y = r[1]
    h = np.zeros((2, 2))
    h[0, 0] = 2 - 400*y + 1200*x*x
    h[0, 1] = - 400*x
    h[1, 0] = h[0, 1]
    h[1, 1] = 200
    return h
