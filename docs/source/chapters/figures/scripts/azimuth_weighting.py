import os
import numpy as np
import lwsspy as lpy
from cartopy.crs import PlateCarree, AzimuthalEquidistant
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

# Create uneven set of stations
lat_eur = 53
lon_eur = 23
lat_us = 0
lon_us = -90
width = 360
height = 180
x, y = lpy.even2Dpoints(100, width, height, 10)
x1, y1 = lpy.even2Dpoints(50, 60, 40, 1)
x2, y2 = lpy.even2Dpoints(50, 50, 120, 3)

# Combine
lat = np.hstack((np.array(y), np.array(y1)+lat_eur, np.array(y2)+lat_us))
lon = np.hstack((np.array(x), np.array(x1)+lon_eur, np.array(x2)+lon_us))


plt.figure(figsize=(10, 3.5))
labels = ["a", "b", "c"]
nbins = 12
for _i in range(3):
    ax = plt.subplot(131 + _i, projection=AzimuthalEquidistant(0, 0))
    ax.set_global()
    lpy.plot_map()

    weights = lpy.azi_weights(0, 0, lat, lon, nbins=nbins*(_i+1))
    plt.scatter(lon, lat, c=weights, cmap='rainbow',
                norm=LogNorm(vmin=min(weights), vmax=max(weights)),
                transform=PlateCarree())
    formatter = ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
    cb = lpy.nice_colorbar(orientation='horizontal', ticks=[0.3, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0],  # np.arange(0.3, 3.0, 0.3),
                           format=formatter, aspect=40, pad=0.05)
    lpy.plot_label(ax, f"{labels[_i]})", location=6, box=False, dist=0.0)
    cb.set_label("Weights")
    plt.title(f"$N_b = {nbins*(_i+1)}$", fontdict=dict(fontsize='small'))
    lpy.plot_label(
        ax,
        f"min: {np.min(weights):3.2f}\n"
        f"max: {np.max(weights):3.2f}\n"
        f"median: {np.median(weights):3.2f}\n",
        location=3, box=False, dist=-0.1, fontdict=dict(fontsize='small'))

plt.subplots_adjust(bottom=0.025, top=0.925, left=0.05, right=0.95)

plt.savefig(os.path.join(lpy.DOCFIGURES, "azimuth_weighting.pdf"))
plt.savefig(os.path.join(lpy.DOCFIGURES, "azimuth_weighting.svg"))
plt.show()
