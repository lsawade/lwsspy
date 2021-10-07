import lwsspy.math as lmat
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_interpolator_1data_point():

    # Create data points
    lat = np.array([10, 15])
    lon = np.array([0, 10])
    data = np.array([3, 4])

    # Create kdtree
    snn = lmat.SphericalNN(lat, lon)

    # Create interpolator
    inter = snn.interpolator(np.array([12.5]), np.array([5]), p=2.0)

    # interpolation
    assert_array_almost_equal(inter(data), np.array([3.50383293]), decimal=4)


def test_interpolator_4data_points():

    # Create data points
    lat = np.array([-10, 10, 10, -10])
    lon = np.array([-10, -10, 10, 10])
    data = np.array([1, 1, 1, 1])

    # Create kdtree
    snn = lmat.SphericalNN(lat, lon)

    # Create interpolator
    inter = snn.interpolator(np.array([0]), np.array([0]))

    # interpolation
    assert_array_almost_equal(inter(data), np.array([1.0]), decimal=4)


def test_interpolator_max_dist():

    # Create data points
    lat = np.array([-10, 10, 10, -10])
    lon = np.array([-10, -10, 10, 10])
    data = np.array([1, 2, 3, 4])

    # Create kdtree
    snn = lmat.SphericalNN(lat, lon)

    # Create interpolator
    inter = snn.interpolator(
        np.array([1]), np.array([1]), maximum_distance=14.0)

    # interpolation
    assert_array_almost_equal(inter(data), np.array([3.0]), decimal=4)


def test_interpolator_k_is_one():

    # Create data points
    lat = np.array([-10, 10, 10, -10])
    lon = np.array([-10, -10, 10, 10])
    data = np.array([1, 2, 3, 4])

    # Create kdtree
    snn = lmat.SphericalNN(lat, lon)

    # Create interpolator
    inter = snn.interpolator(
        np.array([1]), np.array([1]), k=1)

    # interpolation
    assert_array_almost_equal(inter(data), np.array([3.0]), decimal=4)


def test_interpolator_no_constraint():

    # Create data points
    lat = np.array([-10, 10, 10, -10])
    lon = np.array([-10, -10, 10, 10])
    data = np.array([1, 1, 2, 2])

    # Create kdtree
    snn = lmat.SphericalNN(lat, lon)

    # Create interpolator
    inter = snn.interpolator(
        np.array([0]), np.array([5]))

    # interpolation
    assert_array_almost_equal(inter(data), np.array([1.6819]), decimal=4)


def test_interpolator_p_is_large():

    # Create data points
    lat = np.array([-10, 10, 10, -10])
    lon = np.array([-10, -10, 10, 10])
    data = np.array([1, 2, 3, 4])

    # Create kdtree
    snn = lmat.SphericalNN(lat, lon)

    # Create interpolator
    inter = snn.interpolator(
        np.array([5]), np.array([5]), p=16)

    # interpolation
    assert_array_almost_equal(inter(data), np.array([3.0]), decimal=4)
