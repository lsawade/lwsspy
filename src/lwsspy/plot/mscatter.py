import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers


def mscatter(x, y, ax=None, m=None, **kw):
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


if __name__ == "__main__":

    N = 40
    x, y, c = np.random.rand(3, N)
    s = np.random.randint(10, 220, size=N)
    m = np.repeat(["o", "s", "D", "*"], N/4)

    fig, ax = plt.subplots()

    scatter = mscatter(x, y, c=c, s=s, m=m, ax=ax)

    plt.show()
