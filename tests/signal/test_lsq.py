
import numpy as np
import lwsspy as lpy


def test_dlna():

    # Generate random data
    np.random.seed(12345)
    d = np.random.random(100)
    s = np.random.random(100)
    # Compute dlna
    lsq = lpy.lsq(d, s)

    # Check if computation is ok.
    assert abs(lsq) <= 1E-12
