import os
import matplotlib.pyplot as plt
from osgeo import gdal, osr
import cartopy.crs as ccrs
import numpy as np

from matplotlib.colors import Normalize
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter
from scipy import ndimage, misc
import lwsspy.maps as lmaps


def read_maloof_dem(filename):

    # Open file using GDAL read only(!)
    raster = gdal.Open(filename, gdal.GA_ReadOnly)

    # Read the data from the raster
    data = raster.ReadAsArray()

    # Dimensions
    dimx = raster.RasterXSize
    dimy = raster.RasterYSize

    # Number of bands
    nbands = raster.RasterCount

    # Metadata for the raster dataset
    meta = raster.GetMetadata()

    # Get extemt from file
    gt = raster.GetGeoTransform()
    extent = (gt[0], gt[0] + raster.RasterXSize * gt[1],
              gt[3] + raster.RasterYSize * gt[5], gt[3])

    # Get UTM zone from extent
    utmzone = int(np.round(((180+np.mean(extent[:2]))/6))) + 2
    print(utmzone)

    # Get UTM projection from
    projection = ccrs.UTM(utmzone)

    # Get Platitude and longitude vectors
    lat = np.linspace(*(extent[2:]), data.shape[0])
    lon = np.linspace(*(extent[:2]), data.shape[1])

    # Get geo transform/extent for plotting
    return lat, lon, data, projection, extent


# Load data file
area = 'nevada'
ftype = 'dem'
filename = f'/Users/lucassawade/Google Drive/RyansDEMs/drone_derived_{ftype}_{area}.tif'
# animationdir = os.path.join(lbase.DOCFIGURES, "dem_animation")

# Get the file
lat, lon, data, projection, extent = read_maloof_dem(filename)
mask_val = np.min(data)

# Plot the tiff
subplot_kw = dict(projection=ccrs.PlateCarree())
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=subplot_kw)

# Plot depending on whether it is the ortho photo or the elevation model
if ftype == 'dem':
    masked_data = np.where(data < mask_val+1, np.nan, data)
    img = ax.imshow(masked_data.T,
                    extent=extent,  vmin=np.nanmin(masked_data),
                    vmax=np.nanmin(masked_data),
                    cmap='rainbow', origin='upper')
    plt.colorbar(img, aspect=30, pad=0.025)
else:
    img = ax.imshow(np.transpose(
        data[:, ::20, ::20], axes=(1, 2, 0)), extent=extent, origin='upper')

ax.set_xlim(extent[:2])
ax.set_ylim(extent[2:])
gl = ax.gridlines(
    zorder=10, draw_labels=True, linestyle='-', linewidth=0.5)
gl.bottom_labels = False
gl.right_labels = False
ax.tick_params()

plt.show()


T = lmaps.TopographyDesign(lat, lon, data)

T.plot()
