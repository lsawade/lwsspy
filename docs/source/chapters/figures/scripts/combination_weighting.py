import os
import lwsspy as lpy
import numpy as np
import matplotlib.pyplot as plt
from cartopy.crs import PlateCarree, AzimuthalEquidistant
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

# Create some different even2D points to create artificial station densities.
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

# Get azimuthal weights
lat0 = 25
lon0 = -50
nbins = 12
azi_weights = lpy.azi_weights(lat0, lon0, lat, lon, nbins=nbins, p=0.5)

# Get Geoweights
gw = lpy.GeoWeights(lat, lon)
_, _, ref, _ = gw.get_condition()
geo_weights = gw.get_weights(ref)

# Compute combined weights
weights = (azi_weights * geo_weights)
weights /= np.sum(weights)/len(weights)


def plot_weights(ax, weights):

    lpy.plot_map()
    plt.scatter(lon, lat, c=weights, cmap="RdBu_r",
                norm=lpy.MidPointLogNorm(vmin=min(weights), vmax=max(weights),
                                         midpoint=1.0),
                edgecolors='k', linewidths=0.5,
                transform=PlateCarree())
    formatter = ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
    cb = lpy.nice_colorbar(orientation='horizontal', ticks=[0.3, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0],  # np.arange(0.3, 3.0, 0.3),
                           format=formatter, aspect=40, pad=0.075)
    cb.set_label("Weights")
    lpy.plot_label(
        ax,
        f"min: {np.min(weights):3.2f}\n"
        f"max: {np.max(weights):3.2f}\n"
        f"mean: {np.mean(weights):3.2f}\n"
        f"median: {np.median(weights):3.2f}\n",
        location=3, box=False, dist=-0.1, fontdict=dict(fontsize='small'))


plt.figure(figsize=(11, 4))
plt.subplots_adjust(bottom=0.025, top=0.925, left=0.05, right=0.95)
ax = plt.subplot(131, projection=AzimuthalEquidistant(
    central_longitude=lon0, central_latitude=lat0))
ax.set_global()
plot_weights(ax, geo_weights)
plt.title("Geographical Weights", fontdict=dict(fontsize='small'))

ax = plt.subplot(132, projection=AzimuthalEquidistant(
    central_longitude=lon0, central_latitude=lat0))
ax.set_global()
plot_weights(ax, azi_weights)
plt.title("Azimuthal Weights", fontdict=dict(fontsize='small'))

ax = plt.subplot(133, projection=AzimuthalEquidistant(
    central_longitude=lon0, central_latitude=lat0))
ax.set_global()
plot_weights(ax, weights)
plt.title("Combined Weights", fontdict=dict(fontsize='small'))


plt.savefig(os.path.join(lpy.DOCFIGURES, "combination_weighting.pdf"))
plt.savefig(os.path.join(lpy.DOCFIGURES, "combination_weighting.svg"))
plt.show()
