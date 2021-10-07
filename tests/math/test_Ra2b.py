import numpy as np
from numpy.testing import assert_array_almost_equal as aae
import lwsspy.math as lmat


def test_Ra2b():

    a = np.array((1, 0, 0))
    b = np.array((0, 1, 0))
    c = np.array((0, 0, 1))

    aae(a, lmat.Ra2b(b, a) @ b)
    aae(b, lmat.Ra2b(a, b) @ a)
    aae(a, lmat.Ra2b(c, a) @ c)
    aae(c, lmat.Ra2b(a, c) @ a)
    aae(b, lmat.Ra2b(c, b) @ c)
    aae(c, lmat.Ra2b(b, c) @ b)
