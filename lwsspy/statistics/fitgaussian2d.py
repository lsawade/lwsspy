# External imports
import numpy as np
from scipy.optimize import curve_fit

# Internal imports
from .gaussian2d import gaussian2d


def fitgaussian2d(x, y, data, p0):
    """Last modified: Lucas Sawade, 2020.09.15 15.30 (lsawade@princeton.edu)
    """

    # Optimize
    popt, pcov = curve_fit(gaussian2d,
                           (x.ravel(), y.ravel()), data.ravel(),
                           p0=p0)

    return popt, pcov


if __name__ == "__main__":

    import matplotlib.pyplot as plt  # type: ignore
    from ..plot_util.updaterc import updaterc  # type: ignore
    from ..plot_util.figcolorbar import figcolorbar  # type: ignore
    from ..plot_util.remove_ticklabels import remove_yticklabels  # type: ignore
    updaterc()

    # actual m: amplitude, x0, yo, sigma_x, sigma_y, theta, offset
    actual = np.array([4, 115, 90, 25, 35, 22.5, 5])

    # Initial guess m0: amplitude, x0, yo, sigma_x, sigma_y, theta, offset
    initial_guess = np.array([0, 100, 100, 20, 40, 0, 7.5])

    # Create requested points
    x = np.linspace(0, 200, 201)
    y = np.linspace(0, 200, 201)
    x, y = np.meshgrid(x, y)

    # Create data
    data = gaussian2d((x, y), *actual)

    guess = gaussian2d((x, y), *initial_guess)

    # Generate noisy data from original data
    data_noisy = data + 0.2 * np.random.normal(size=data.shape)

    # Optimize
    popt, pcov = fitgaussian2d(x, y, data_noisy, initial_guess)
    inverted_data = gaussian2d((x, y), *popt)

    # Plotting
    vmin, vmax = (5, 9)

    titles = ['Data', r'Data + $\epsilon$', r'Initial Guess',
              r'Inverted Distribution (Contour)']
    datas = [data, data_noisy, guess, inverted_data]

    # plot gaussian2d data generated above
    axes = []
    fig = plt.figure(figsize=(10, 3.5))
    counter = 141

    for title, data in zip(titles, datas):
        axes.append(plt.subplot(counter))
        plt.pcolormesh(data, edgecolor=None, vmin=vmin, vmax=vmax, 
                       zorder=-15)
        plt.axis([0, 200, 0, 200])
        plt.xlabel('x')
        plt.title(title)
        plt.gca().set_aspect('equal', 'box')

        if counter > 141:
            remove_yticklabels(plt.gca())
        else:
            plt.ylabel('y')
        counter += 1

    # Create colorbar with dict for all four plots
    cbar_dict = {"orientation": "horizontal",
                 "pad": 0.175,
                 "aspect": 40,
                 "shrink": 0.4
                 }
    c = figcolorbar(fig, axes, vmin=vmin, vmax=vmax, **cbar_dict)

    plt.show()