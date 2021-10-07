import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class PlotZoom:
    def __init__(self, ax: Axes, zax: Axes,
                 x: float, y: float, winx: float, winy: float):
        """Uses two axes, a starting point and window size to create zoom
        window of the picking plot.

        Parameters
        ----------
        ax : matplotlib axes
            original axes
        zax : matplotlib axes
            destination axes for the zooming
        x : float
            start x center of zoom window
        y : float
            sttart y center of zoom window
        winx : float
            x window size in +/-
        winy : float
            y window size in +/-


        Notes
        -----

        .. note::

            You have to plot everything into the destination axes to see
            the zoomed in plots!

        """

        self.ax = ax
        self.zax = zax
        self.x = x
        self.y = y
        self.winx = winx
        self.winy = winy
        self.press = None
        self.center, = plt.plot(x, y, 'b+', markersize=100)

    def connect(self):
        """connect to all the events we need"""
        self.cidmotion = self.ax.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_motion(self, event):
        """on motion the zoom window is updated."""
        if (event.xdata is not None) and (event.ydata is not None):
            self.x, self.y = (event.xdata, event.ydata)
        else:
            return
        # Change center and window limits
        self.center.set_data(self.x, self.y)
        self.zax.set_xlim(self.x - self.winx, self.x + self.winx)
        self.zax.set_ylim(self.y + self.winy, self.y - self.winy)

        # Draw
        self.zax.figure.canvas.draw()

    def disconnect(self):
        """Disconnect"""
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)
