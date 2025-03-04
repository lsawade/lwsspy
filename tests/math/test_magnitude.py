
import lwsspy.math as lmat
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_magnitude():

    assert lmat.magnitude(230.03424235) == 2
    assert lmat.magnitude(20.32) == 1
    assert lmat.magnitude(0.0023) == -3
    assert lmat.magnitude(2302342) == 6
    assert lmat.magnitude(0) == 0
    assert_array_almost_equal(lmat.magnitude((245, 0, 0.00004)), (2, 0, -5))
