Tips And Tricks for Matplotlib
==============================

This page is not really a part of the documentation of the package, but rather
some tips and tricks I have gathered over time.


Partially rasterizing your PDF output
+++++++++++++++++++++++++++++++++++++

You probably have tried outputting your meshplot in matplotlib and wondered
why the heck it is taking soo long! 
The explanation is simple. Every coordinate and data combination is creating a 
box (with coordinattes) with a color that has to be output and written to 
the PDF file.
When your figure includes multiple meshplots with 1000x1000 plots, 
the number of boxes becomes very large and the number of colors/coordinates even
larger.
One way to get around it, is simply saving it as `png` file.
That however doesn't make your plots exactly publishable.

A workaround is partially rasterizing your plots. That can be done with the
following command:

.. literalinclude:: figures/scripts/rasterize.py
  :language: python


It works for both `.svg` and `.pdf`. Probably others too, but I haven't tried.

Below the image produced by the code above

.. image:: figures/test_rasterize.svg



Make x/y labels invisible on shared axes plots
++++++++++++++++++++++++++++++++++++++++++++++

To make plots with subfigures more beautiful, you may want to remove axes
labels if the plots share the axes!

.. literalinclude:: figures/scripts/remove_labels.py
  :language: python

.. image:: figures/remove_labels.svg