import matplotlib.pyplot as plt
import numpy as np


def animatevolume(file, direction='x', label=False):

    # File
    var = np.load(file)

    # Load variables
    x = var['x']
    y = var['y']
    z = var['z']
    V = var['V']
    extent = [np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)]

    if label:
        N = 2
        L = var['label']
        figsize = (12, 6)
        bmin = L.min()
        bmax = L.max()
    else:
        N = 1
        figsize = (7, 6)

    plt.figure(figsize=figsize)
    plt.subplot(1, N, 1)
    vmin = np.quantile(V, 0.025)
    vmax = np.quantile(V, 0.975)

    if direction == 'x':
        im = plt.imshow(
            V[0, :, :].T, aspect='auto',
            extent=(*extent[2:4], *extent[4:][::-1]), rasterized=True,
            vmin=vmin, vmax=vmax, interpolation='none')
        M = len(x)
        plt.suptitle(f'x: {x[0]}')
    elif direction == 'y':
        im = plt.imshow(
            V[:, 0, :].T, aspect='auto',
            extent=(*extent[:2], *extent[4:][::-1]), rasterized=True,
            vmin=vmin, vmax=vmax, interpolation='none')
        M = len(y)
        plt.suptitle(f'y: {y[0]}')
    else:
        raise ValueError(f'Direction {direction} not implemented.')

    if label:
        plt.subplot(1, N, 2)
        if direction == 'x':
            lim = plt.imshow(
                V[0, :, :].T, aspect='auto',
                extent=(*extent[2:4], *extent[4:][::-1]), rasterized=True,
                vmin=bmin, vmax=bmax, interpolation='none')
        elif direction == 'y':
            lim = plt.imshow(
                V[:, 0, :].T, aspect='auto',
                extent=(*extent[:2], *extent[4:][::-1]),
                vmin=bmin, vmax=bmax, interpolation='none')

    plt.show(block=False)

    # Start from one since first index was drawn already
    for i in range(1, M):
        if direction == 'x':
            im.set_data(V[i, :, :].T)
            plt.suptitle(f'x: {x[i]}')
        elif direction == 'y':
            im.set_data(V[:, i, :].T)
            plt.suptitle(f'y: {y[i]}')

        if label:
            if direction == 'x':
                lim.set_data(L[i, :, :].T)
            elif direction == 'y':
                lim.set_data(L[:, i, :].T)

        plt.draw()
        plt.pause(0.0001)


if __name__ == '__main__':

    import sys

    if len(sys.argv) == 3:
        filename, direction = sys.argv[1:]
        label = False
    elif len(sys.argv) == 4:
        filename, direction, label = sys.argv[1:]

    animatevolume(filename, direction=direction, label=bool(label))
