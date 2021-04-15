import matplotlib.pyplot as plt
from osgeo import gdal, osr
import cartopy.crs as ccrs
import numpy as np
import lwsspy as lpy
from matplotlib.colors import Normalize
from scipy.signal.windows import tukey

# Load data file
filename = '/Users/lucassawade/Google Drive/RyansDEMs/drone_derived_dem_yukon.tif'
raster = gdal.Open(filename, gdal.GA_ReadOnly)

# Read the data from the raster
data = raster.ReadAsArray()

# Check type of the variable 'raster'
type(raster)

# Dimensions
dimx = raster.RasterXSize
dimy = raster.RasterYSize

# Number of bands
nbands = raster.RasterCount

# Metadata for the raster dataset
meta = raster.GetMetadata()

# Find UTM
gt = raster.GetGeoTransform()
extent = (gt[0], gt[0] + raster.RasterXSize * gt[1],
          gt[3] + raster.RasterYSize * gt[5], gt[3])
utmzone = int(np.round(((180+np.mean(extent[:2]))/6)))

# Projection
proj = raster.GetProjection()
srs = osr.SpatialReference(wkt=proj)


# Get Cartopy projection from PROJCS
projcs = srs.GetAuthorityCode('PROJCS')
projection = ccrs.UTM(utmzone)

# Get geo transform/extent for plotting
mindata = np.min(data)
ndata = np.where(data == mindata, np.nan, data)
mdata = np.ma.masked_invalid(ndata)


datamin = np.nanmin(ndata)
datamax = np.nanmax(ndata)

# Plot the tiff
subplot_kw = dict(projection=projection)
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=subplot_kw)

img = ax.imshow(data, extent=extent, vmin=datamin, vmax=datamax,
                origin='upper', cmap='rainbow')


x = np.linspace(*(extent[:2]), data.shape[1])
y = np.linspace(*(extent[2:]), data.shape[0])


fc = 'k'
fig = plt.figure(facecolor=fc, figsize=(16, 9))
ax = plt.axes(facecolor=fc)

# Get taper for edge ffade
taper = tukey(len(x), alpha=0.25)

# Normalize
norm = Normalize(vmin=datamin, vmax=datamax)

# Normalize the topography for plotting the line
normtopo = norm(ndata)

# Loop over lines
ny = 30
dy = y[1]-y[0]
exagy = 50
for _i, _ilat in enumerate(y[::ny]):

    # Get the data
    yt = _ilat * np.ones_like(x) + normtopo[_i * ny, :] * dy * ny * exagy
    z = ndata[_i * ny, :]
    linewidths = normtopo[_i * ny, :] * taper
    baseline = np.min(y) - (dy * exagy)

    # Plot polygons to lines in the back aren't visible
    plt.fill_between(x, yt, y2=baseline, facecolor=fc,
                     edgecolor='none', zorder=-_i)

    # Plot lines with linewidth and topo cmap
    lines, sm = lpy.plot_xyz_line(
        x, yt, z, cmap="rainbow", norm=norm,
        capstyle='round', linewidths=linewidths, zorder=-_i-0.5,
        clip_on=False)

ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y) + dy * ny * exagy)
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)  # 1.0+yrat)

plt.savefig(os.path.join(lpy.DOCFIGURES, "yukon_dem.png"), dpi=300)
