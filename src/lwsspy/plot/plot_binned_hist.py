
import matplotlib.pyplot as plt


def plot_binned_hist(bins, counts, *args, **kwargs):

    # Get centroids
    centroids = (bins[1:] + bins[:-1]) / 2

    # Number of bins
    Nb = len(counts)
    brange = (min(bins), max(bins))

    return plt.hist(
        centroids, *args, bins=Nb, weights=counts, range=brange, **kwargs)
