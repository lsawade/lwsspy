import os
import matplotlib.pyplot as plt
from cartopy.crs import PlateCarree
# Internal
import lwsspy.maps as lmaps
import lwsspy.plot as lplt
import lwsspy.base as lbase
lplt.updaterc()

mapextent = [-15, 30, 30, 74]


plt.figure()
ax = plt.axes(projection=PlateCarree())
lmaps.plot_map(fill=False)
ax.set_extent(mapextent)
lmaps.plot_topography(extent=mapextent, subsampling=1, sigma=0.0)
ax.set_rasterization_zorder(-10)  # Important line!3
# Save figure with rasterization with 300dpi
plt.savefig(os.path.join(lbase.DOCFIGURES, "topography_europe.svg"), dpi=150)


mapextent = [-180, 180, -90, 90]
plt.figure(figsize=(7.5, 3.75))
ax = plt.axes(projection=PlateCarree())
lmaps.plot_map(fill=False)
ax.set_extent(mapextent)
lmaps.plot_topography(extent=mapextent, subsampling=4,
                      sigma=2.0, colorbar=False)
ax.set_rasterization_zorder(-10)  # Important line!3
# Save figure with rasterization with 300dpi
plt.savefig(os.path.join(lbase.DOCFIGURES, "topography_earth.svg"), dpi=150)
plt.show()
