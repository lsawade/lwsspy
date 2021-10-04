from typing import Union, Tuple
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import lwsspy as lpy
from .plot_zoom import PlotZoom


def pick_data_from_image(infile: str, outfile: Union[str, None] = None,
                         extent=[0, 1, 0, 1], logx: bool = False,
                         logy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function's purpose is the replication of previously published data.
    The function ttakes in an image filename containing a graph that is
    cropped to axes boundaries and the data extent. One can then pick the data
    from the graph in the image and it will automatically rescale the picked
    pixel values to data domain.

    Optionally the data can be save to either npy or csv depending on
    extension. If the oufile's extension is ``.npy``, the data will be saved
    as a single ``numpy.ndarray`` with two columns for the(x, y) vectors.
    If the file ending is ``.csv``, the ``csv`` file would be have the format:

    ::

        x0, y0
        x1, y1
        ...

    Parameters
    ----------
    infile: str
        Image file name to be loaded by the function. The image should be
        cropped to the axes edges.
    outfile: Union[str, None], optional
        Optional output file name depending on filename ending either a
        npy array with two columns(x, y) will be stored or a csv file,
        by default None
    extent: list, optional
        extent of the cropped axes in data units. Format: [xi, xf, yi, yf],
        by default[0, 1, 0, 1]
    logx: bool, optional
        Flag to say whether the x axis in the image is in logarithmic scale,
        by default False
    logy: bool, optional
        Flag to say whether the y axis in the image is in logarithmic scale,
        by default False


    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x, y vectors of the picked data.

    Raises
    ------
    ValueError
        if file ending is not ``.csv`` or ``.npy``


    See Also
    --------
    lwsspy.utils.pixels2data.pixels2data : Scaling from pixels to dataunits


    Examples
    --------

    For an image with data in linearscale and axes data axis limits of 
    ``[0, 1, 0, 1]`` ``x`` and ``y`` can be picked and saved as follows.

    >>> import lwsspy as lpy
    >>> x,y = lpy.pick_data_from_image('test.png', 'testdata.csv')

    Change ``.csv`` to  ``.npy`` to save numpy binary file.


    Notes
    -----

    .. note::

        Picking Controls

        - Click to add points

        - Press Delete or Backspace to delete points

        - Press Enter to finish

    .. warning::

        For semilog conversions, the extent of the axis cannot be  0 !!
        (log(0) issue)


    .. note::

        Note for further development: A better way of doing this would be to 
        actually integrate the picker and the zoom window. Meaning, a while loop
        for ginput until an enter is recorded.
        Connect callbacks for backspace, enter, and mouseclicks

        Backspace - pops the last item and removes the last scatter point as long
        as the point list is not empty

        Enter - Finishes the selection

        Mouse - Click adds points

        This way one can plot the selected points in both the zoom window AND
        and the main window.

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.06 11.00

    """

    if outfile is not None:
        fileending = outfile[-3:]
        if fileending not in ["npy", "csv"]:
            raise ValueError(
                f"Wrong file format. Expected 'npy' or 'csv'. "
                f"Gotten: '{fileending}'")

    # Load image
    im = mpimg.imread(infile)
    imagextent = [0, im.shape[1], 0, im.shape[0]]
    xc = np.round(im.shape[1] * 0.5)
    yc = np.round(im.shape[0] * 0.5)
    xwin = im.shape[1] * 0.025
    ywin = im.shape[0] * 0.025

    # Create and show image
    lpy.plot.pz_figure()
    ax = plt.axes()
    plt.imshow(im)
    tit = plt.title("Press any key to start picking values...")
    plt.show(block=False)

    # Wati for the User to press a key to start
    plt.waitforbuttonpress()

    # Create zoom window
    plt.figure()
    zax = plt.axes()
    zax.imshow(im)
    zax.set_xlim((xc - xwin, xc + xwin))
    zax.set_ylim((yc + ywin, yc - ywin))
    PZ = PlotZoom(ax, zax, xc, yc, xwin, ywin)
    PZ.connect()
    plt.show(block=False)

    # Change title so that the user can start putting in values
    tit.set_text(
        "Pick values in zoom window. Delete a point using backspace, finish with Enter")
    ax.figure.canvas.draw()

    # Use matplotlib.pyplot.ginput for easy data picking
    plt.sca(ax)  # Set current aces to pick points
    xy = plt.ginput(n=-1, timeout=-1, show_clicks=True)
    # PZ.disconnect()

    # Convert list of points to two vectors
    xpx = np.array([p[0] for p in xy])
    ypx = np.array([p[1] for p in xy])

    # Plot Values into the main window
    zax.plot(xpx, ypx, "b+")

    # Convert Pixel to data
    x = lpy.utils.pixels2data(xpx, imagextent[0], imagextent[1],
                              extent[0], extent[1], log=logx)
    y = lpy.utils.pixels2data(ypx, imagextent[3], imagextent[2],
                              extent[2], extent[3], log=logy)

    # Output data to file of outfile is not Nonne
    if outfile is not None:

        points = np.vstack((x, y)).T
        if fileending == 'npy':
            np.save(outfile, points)
        else:
            # Header
            header = f"Data picked from file: {infile}"
            np.savetxt(outfile, points, delimiter=',', header=header)

    else:
        plt.figure()
        plt.plot(x, y)
        if logx:
            plt.xscale('log')
        if logy:
            plt.yscale('log')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return x, y


def bin():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='infile',
                        help='input image',
                        required=True, type=str)
    parser.add_argument('-o', dest='outfile',
                        help='Output filename', default=None,
                        required=False, type=str)
    parser.add_argument('-xi', dest='xmin',
                        help='xmin', default=0.0,
                        required=False, type=float)
    parser.add_argument('-xf', dest='xmax',
                        help='xmax', default=1.0,
                        required=False, type=float)
    parser.add_argument('-yi', dest='ymin',
                        help='ymin', default=0.0,
                        required=False, type=float)
    parser.add_argument('-yf', dest='ymax',
                        help='xmax', default=0.0,
                        required=False, type=float)
    parser.add_argument('-logx', dest='logx',
                        help='Flag to make specify that x axis is logarithmic',
                        action='store_true', default=False, required=False)
    parser.add_argument('-logy', dest='logy',
                        help='Flag to make specify that y axis is logarithmic',
                        action='store_true', default=False, required=False)

    args = parser.parse_args()

    # Run
    pick_data_from_image(args.infile, args.outfile,
                         extent=[args.xmin, args.xmax, args.ymin, args.ymax],
                         logx=args.logx, logy=args.logy)
