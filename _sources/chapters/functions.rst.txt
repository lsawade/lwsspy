Aaaalll the functions!
======================

Inversion
+++++++++

.. autoclass:: lwsspy.inversion.optimizer.Optimization
    :members: __init__

.. figure:: figures/optimization.svg

    Simple test of optimizing the Rosenbrock function.

.. figure:: figures/optimization_x4.svg

    Simple test of optimizing a 1D function. Note that we have 2 local
    minima which of course is a problem for local minimization schemes.


Maps
++++

.. autofunction:: lwsspy.maps.fix_map_extent.fix_map_extent

.. autofunction:: lwsspy.maps.plot_litho.plot_litho

.. autofunction:: lwsspy.maps.plot_map.plot_map

.. autofunction:: lwsspy.maps.plot_topography.plot_topography

.. figure:: figures/topography_europe.svg

    Shows the topography of Europe using Etopo1.

.. figure:: figures/topography_earth.svg

    Shows the topography of the Earth using Etopo1. Note: Ice Sheets
    have not been implemented yet.

.. autofunction:: lwsspy.maps.read_etopo.read_etopo

.. autofunction:: lwsspy.maps.read_litho.read_litho

.. autofunction:: lwsspy.maps.topocolormap.topocolormap



Math
++++

Coordinate Transformations
--------------------------

.. autofunction:: lwsspy.math.cart2geo.cart2geo

.. autofunction:: lwsspy.math.cart2pol.cart2pol

.. autofunction:: lwsspy.math.cart2sph.cart2sph

.. autofunction:: lwsspy.math.geo2cart.geo2cart

.. autofunction:: lwsspy.math.pol2cart.pol2cart

.. autofunction:: lwsspy.math.project2D.project2D

.. autofunction:: lwsspy.math.rotation_matrix.rotation_matrix

.. autofunction:: lwsspy.math.sph2cart.sph2cart

.. autofunction:: lwsspy.math.rodrigues.rodrigues

.. autofunction:: lwsspy.math.Ra2b.Ra2b




Miscellaneous
-------------

.. autofunction:: lwsspy.math.convm.convm

.. autofunction:: lwsspy.math.eigsort.eigsort

.. autofunction:: lwsspy.math.logistic.logistic

.. autofunction:: lwsspy.math.magnitude.magnitude

.. autoclass:: lwsspy.math.SphericalNN.SphericalNN
    :members:

Plotting Utilities
++++++++++++++++++

.. autofunction:: lwsspy.plot.figcolorbar.figcolorbar

.. autoclass:: lwsspy.plot.fixedpointcolornorm.FixedPointColorNorm

.. autofunction:: lwsspy.plot.nice_colorbar.nice_colorbar

.. autofunction:: lwsspy.plot.pick_data_from_image.pick_data_from_image

.. autofunction:: lwsspy.plot.plot_label.plot_label

.. autofunction:: lwsspy.plot.remove_ticklabels.remove_xticklabels

.. autofunction:: lwsspy.plot.remove_ticklabels.remove_yticklabels

.. autofunction:: lwsspy.plot.updaterc.updaterc

.. autofunction:: lwsspy.plot.view_colormap.view_colormap


Statistics
++++++++++

.. autofunction:: lwsspy.statistics.fakerelation.fakerelation

.. figure:: figures/modelled_covarying_dataset.svg

    Modeled covarying data sets.

.. autofunction:: lwsspy.statistics.errorellipse.errorellipse

.. figure:: figures/error_ellipse.svg

    Generating the error ellipse for a covarying dataset.

.. autofunction:: lwsspy.statistics.gaussian.gaussian

.. autofunction:: lwsspy.statistics.gaussian2d.gaussian2d

.. figure:: figures/gaussian2d.svg

    Forward modelling a 2D Gaussian distribution.

.. autofunction:: lwsspy.statistics.fitgaussian2d.fitgaussian2d

.. figure:: figures/fitgaussian2d.svg

.. autofunction:: lwsspy.statistics.distlist.distlist

.. figure:: figures/distlist.svg

    Creating a list of distributions.

.. autofunction:: lwsspy.statistics.clm.clm

.. figure:: figures/clm.svg

    Showing that the central limit theorem holds. Note that the convolution
    limits are not actually correct, and this plot is solely for illustration.

.. autofunction:: lwsspy.statistics.plot_binnedstats.plot_binnedstats

.. figure:: figures/clm.svg

    Showing that the central limit theorem holds. Note that the convolution
    limits are not actually correct, and this plot is solely for illustration.



Utilities
+++++++++

I/O
---

.. automodule:: lwsspy.utils.io
    :members:

Print Utilities
---------------

.. automodule:: lwsspy.utils.output
    :members:

Miscellaneous
-------------

.. autofunction:: lwsspy.utils.chunks.chunks

.. autofunction:: lwsspy.utils.cpu_count.cpu_count

.. autofunction:: lwsspy.utils.pixels2data.pixels2data

.. automodule:: lwsspy.utils.threadwork
    :members:

Weather
+++++++

.. autofunction:: lwsspy.weather.drop2pickle.drop2pickle

.. autoclass:: lwsspy.weather.requestweather.requestweather

.. autoclass:: lwsspy.weather.weather.weather
