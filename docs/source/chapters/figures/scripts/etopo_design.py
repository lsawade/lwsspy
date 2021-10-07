import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import patches
from scipy.ndimage import gaussian_filter
from scipy.signal.windows import tukey
from scipy.interpolate import interpn
from cartopy.crs import PlateCarree
import cartopy.crs as ccrs
import lwsspy.maps as lmaps
import lwsspy.plot as lplt
import lwsspy.base as lbase

# Get topography
etopo = lmaps.read_etopo()
etopodata = etopo.bedrock.data
etopocmap = lmaps.topocolormap()
etoponorm = lplt.FixedPointColorNorm(vmin=-10000, vmax=8000)

# Get subset
latx = 39.15, 43.96
lonx = -123.08, -112.03
ltopo = etopo.sel(latitude=slice(*latx), longitude=slice(*lonx))
lat = ltopo.latitude.data
lon = ltopo.longitude.data
data = ltopo.bedrock.data

# Interpolation setup
mult = 3
minlon, maxlon = np.min(lon), np.max(lon)
minlat, maxlat = np.min(lat), np.max(lat)
nlat = np.linspace(minlat, maxlat, 3*len(lat))
nlon = np.linspace(minlon, maxlon, 3*len(lon))
mlat, mlon = np.meshgrid(nlat, nlon)

# Interpolate
ndata = interpn((lat, lon), data, (mlat.T, mlon.T))

# Filter to create smooth lines (etopo cann be jagged if close)
fdata = gaussian_filter(ndata, 15.0)

# Normalize
norm = Normalize(vmin=np.min(fdata), vmax=np.max(fdata))

# Normalize the topography for plotting the line
normtopo = norm(fdata)

# Plotting the map that actually shows the data:
pad = 0.025
w = 0.14
h = 0.2

fig = plt.figure()
ax = plt.gca()

plt.imshow(ndata[::-1, :], extent=(minlon, maxlon, minlat, maxlat),
           cmap=etopocmap, norm=etoponorm, aspect='auto')

# Plot axis inset
pad = 0.025
w = 0.14
h = 0.2
a = ax.get_position()
iax_pos = [a.x1-(w+pad)*a.width, a.y1-(h+pad) *
           a.height, w*a.width, h*a.height]
iax = fig.add_axes(iax_pos, projection=PlateCarree())
iax.axis('on')
lmaps.plot_map()
iax.set_xlim(-130, -110)
iax.set_ylim(30, 50)

iax.imshow(etopodata[::-5, ::5], extent=(-179.99, 179.99, -89.99, 89.99),
           cmap=etopocmap, norm=etoponorm, aspect='auto')
rect = patches.Rectangle((minlon, minlat), (maxlon-minlon), (maxlat-minlat),
                         linewidth=1, edgecolor='r', facecolor='none')
iax.add_patch(rect)

lplt.remove_ticklabels(iax)
lplt.remove_ticks(iax)
plt.savefig(os.path.join(lbase.DOCFIGURES, "etopo_design_map.png"),
            transparent=True, dpi=300)

# Defining the step
nx = 15
ny = 10
dx = lon[nx] - lon[0]
dy = lat[nx] - lat[0]

# Defining exageration
exagx = 10
exagy = 5

# Get ratio
yrat = np.abs((exagy*dy)/(maxlat-minlat))
xrat = np.abs((exagx*dx)/(maxlon-minlon))


# plt.subplot(312)
fc = 'k'
fig = plt.figure(facecolor=fc, figsize=(16, 9))
ax = plt.axes(facecolor=fc)
plt.subplots_adjust(left=0.0, right=1.0,
                    bottom=0.0, top=1.0)  # 1.0+yrat)

# Create linewidth taper for a fade
taper = tukey(len(nlon), alpha=0.25)
for _i, _ilat in enumerate(nlat[::ny]):

    # Get the data
    x = nlon
    y = _ilat * np.ones_like(nlon) + normtopo[_i * ny, :] * dy * exagy
    z = fdata[_i * ny, :]
    linewidths = normtopo[_i * ny, :]*taper
    baseline = minlat-(dy * exagy)

    # Plot polygons to lines in the back aren't visible
    plt.fill_between(x, y, y2=baseline, facecolor=fc,
                     edgecolor='none', zorder=-_i)

    # Plot lines with linewidth and topo cmap
    lines, sm = lplt.plot_xyz_line(
        x, y, z, cmap="rainbow", norm=norm,
        capstyle='round', linewidths=linewidths, zorder=-_i-0.5,
        clip_on=False)
ax.set_xlim(minlon, maxlon)
ax.set_ylim(minlat, maxlat + dy * exagy)


plt.savefig(os.path.join(lbase.DOCFIGURES, "etopo_design_lat.png"), dpi=300)


fc = 'k'
fig = plt.figure(facecolor=fc, figsize=(16, 9))
ax = plt.axes(facecolor=fc)

# Get taper for edge ffade
taper = tukey(len(nlat), alpha=0.25)

# Loop over lines
for _i, _ilon in enumerate(nlon[::nx]):

    # Get the data
    x = nlat[::-1]
    y = _ilon * np.ones_like(nlat) + normtopo[:, _i * nx] * dx * exagx
    z = fdata[:, _i * nx]
    linewidths = normtopo[:, _i * nx]*taper
    baseline = minlon-(dx * exagx)

    # Plot polygons to lines in the back aren't visible
    plt.fill_between(x, y, y2=baseline, facecolor=fc,
                     edgecolor='none', zorder=-_i)

    # Plot lines with linewidth and topo cmap
    lines, sm = lplt.plot_xyz_line(
        x, y, z, cmap="rainbow", norm=norm,
        capstyle='round', linewidths=linewidths, zorder=-_i-0.5,
        clip_on=False)

ax.set_xlim(minlat, maxlat)
ax.set_ylim(minlon, maxlon + dx * exagx)
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)  # 1.0+yrat)

plt.savefig(os.path.join(lbase.DOCFIGURES, "etopo_design_lon.png"), dpi=300)
