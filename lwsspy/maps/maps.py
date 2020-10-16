import matplotlib.pyplot as plt
import cartopy
import numpy as np
## 
plt.figure()


mu, sigma = 0, 0.5 # mean and standard deviation
lat = np.random.normal(mu, sigma, 10000) * 90.0
lon = np.random.normal(mu, sigma, 10000) * 180.0
proj = cartopy.crs.EqualEarth()

ax = plt.axes(projection=proj)
ax.coastlines(resolution='110m')
ax.gridlines()

ax.plot(lon, lat, '.')
ax.set_extent([-180,180,-90,90])
# hex_bin = ax.hexbin(lon, lat, C=None, cmap="Wistia", alpha=0.5)
# cbar = plt.colorbar(hex_bin, orientation='vertical')

plt.show(block=True)
