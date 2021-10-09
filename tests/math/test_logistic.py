from numpy.testing import assert_array_almost_equal as aae
import lwsspy.math as lmat


def test_logistic():

    # Check k
    aae(lmat.logistic(10.0, k=5), 1.0, decimal=5)
    aae(lmat.logistic(-10.0, k=5), 0.0, decimal=5)

    # Check t and b
    aae(lmat.logistic(5, k=2.0), 1.0)
    aae(lmat.logistic(0.0, k=2.0, t=8.0, b=1.0), 9.0/2, decimal=5)
    aae(lmat.logistic(4.0, k=2.0, t=8.0, b=1.0), 8.0, decimal=4)
    aae(lmat.logistic(-4.0, k=2.0, t=8.0, b=1.0), 1.0, decimal=4)

    # Check x0
    aae(lmat.logistic(5, k=2.0), 1.0)
    aae(lmat.logistic(1.0, x0=1.0, k=2.0, t=8.0, b=1.0), 9.0/2, decimal=5)
    aae(lmat.logistic(3.0, x0=-1.0, k=2.0, t=8.0, b=1.0), 8.0, decimal=4)
    aae(lmat.logistic(-3.0, x0=0.5, k=2.0, t=8.0, b=1.0), 1.0, decimal=4)
