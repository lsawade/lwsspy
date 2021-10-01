import os
import numpy as np


def gaussian2d(xdata, amplitude: float = 1, xo: float = 0, yo: float = 0,
               sigma_x: float = 1, sigma_y: float = 1,
               theta: float = 0, offset: float = 0):
    """Generates arbitrary gaussian distribution for the corresponding arrays
    x and, at expected value (xo, yo), and standard deviations
    (sigma_x, sigma_y). To ensure a possibilty of angled distributions
    and offset distributions model parameters theta and offset are included.

    Args:
        x (np.ndarray):
            x coordinates
        y (np.ndarray):
            y coordinates
        amplitude (float):
            amplitude of the Gaussian function
        xo (float):
            center in x direction
        yo (float):
            center in y direction
        sigma_x ():
            std in x direction
        sigma_y (float):
            std in y direction
        theta (float):
            angle of the distribution
        offset (float):
            offset of the distribution (could be gaussian + offset)

    Returns:
        G(x, y) as np.ndarray

    Last modified: Lucas Sawade, 2020.09.15 19.44 (lsawade@princeton.edu)
    """

    # Get x and y
    (x, y) = xdata

    # Compute coefficient
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return offset + amplitude * np.exp(-(a*((x-xo)**2)
                                         + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from ..plot.updaterc import updaterc
    from .. import DOCFIGURES
    updaterc()

    # Create requested points
    x = np.linspace(0, 200, 201)
    y = np.linspace(0, 200, 201)
    x, y = np.meshgrid(x, y)

    # Create data
    data = gaussian2d(xdata=(x, y), amplitude=3, xo=100, yo=100,
                      sigma_x=20, sigma_y=40, theta=25, offset=10)

    # plot gaussian2d data generated above
    plt.figure()
    plt.pcolormesh(data, edgecolor=None, zorder=-15)
    plt.gca().set_rasterization_zorder(-10)
    plt.axis([0, 200, 0, 200])
    plt.colorbar()
    plt.savefig(os.path.join(DOCFIGURES, 'gaussian2d.svg'), dpi=300)
    plt.show()
