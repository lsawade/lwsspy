import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import lwsspy.maps
import lwsspy.base

# Create random artificial track
x = np.linspace(0, 2, 10)
y = x**2 + np.sin(10*x) * 2 * np.exp(-x**2)
lon = x*180 - 180
lat = y*20 - 10

# Get equidistant points along great circles tracks of waypoints
edist = 2.0  # distance between points
qlat, qlon, _ = lwsspy.maps.gctrack(lat, lon, edist)

# Plot stuff
plt.figure(figsize=(8, 4.5))
ax = lwsspy.maps.map_axes("carr")
ax.set_global()
lwsspy.maps.plot_map()
plt.plot(lon, lat, color='k',  transform=ccrs.PlateCarree())
plt.plot(qlon, qlat, color='r',  transform=ccrs.PlateCarree())

plt.savefig(os.path.join(lwsspy.base.DOCFIGURES,
                         "gctrack.svg"), transparent=True)
