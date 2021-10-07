
import numpy as np
import lwsspy.signal as lsig


def test_norm2():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])

    # Compute dlna
    norm = lsig.norm2(d)

    # Check if computation is ok.
    assert (norm-14.0) <= 1E-12


def test_norm1():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])

    # Compute dlna
    norm = lsig.norm1(d)

    # Check if computation is ok.
    assert (norm-6.0) <= 1E-12


def test_dnorm2():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])
    s = np.array([4, 5, 6])

    # Compute dlna
    norm = lsig.dnorm2(d, s)

    # Check if computation is ok.
    assert (norm-13.5) <= 1E-12


def test_dnorm1():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])
    s = np.array([4, 5, 6])

    # Compute dlna
    norm = lsig.dnorm1(d, s)

    # Check if computation is ok.
    assert (norm-9.0) <= 1E-12
