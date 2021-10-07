import numpy as np
import lwsspy.signal as lsig


def test_dlna():

    # Generate random data
    np.random.seed(12345)
    d = np.random.random(100)

    # Compute dlna
    dlnA = lsig.dlna(d, d)

    # Check if computation is ok.
    assert abs(dlnA) <= 1E-12
