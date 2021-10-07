# %%
import numpy as np
import matplotlib.pyplot as plt
import cartopy
from cartopy.crs import PlateCarree
from matplotlib.colors import Normalize

# %%


def plotmap():

    ax = plt.gca()
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='None',
                   linewidth=0.5, facecolor=(0.8, 0.8, 0.8))
    ax.spines['geo'].set_linewidth(0.75)


def make2dhist(lon, lat, z, latbins, lonbins):
    '''Takes the inputs and creates 2D histogram'''
    zz, _, _ = np.histogram2d(lon, lat, bins=(
        lonbins, latbins), weights=z, normed=False)
    counts, _, _ = np.histogram2d(lon, lat, bins=(lonbins, latbins))\

    # Workaround for zero count values tto not get an error.
    # Where counts == 0, zi = 0, else zi = zz/counts
    zi = np.zeros_like(zz)
    zi[counts.astype(bool)] = zz[counts.astype(bool)] / \
        counts[counts.astype(bool)]
    zi = np.ma.masked_equal(zi, 0)

    return lonbins, latbins, zi


def make3dhist(lon, lat, t, z, latbins, lonbins, tbins):
    '''Takes the inputs and creates 3D histogram just as the 2D histogram
    function'''
    zz, _ = np.histogramdd(
        np.vstack((lon, lat, t)).T,
        bins=(lonbins, latbins, tbins),
        weights=z, normed=False)

    counts, _ = np.histogramdd(
        np.vstack((lon, lat, t)).T,
        bins=(lonbins, latbins, tbins))

    # Workaround for zero count values tto not get an error.
    # Where counts == 0, zi = 0, else zi = zz/counts
    zi = np.zeros_like(zz)
    zi[counts.astype(bool)] = zz[counts.astype(bool)] / \
        counts[counts.astype(bool)]
    zi = np.ma.masked_equal(zi, 0)

    return lonbins, latbins, tbins, zi


# %% 2D Case based on your example
def create2Ddata():
    '''Makes some random data'''

    N = 2000
    lat = 10 * np.random.rand(N) + 40
    lon = 25 * np.random.rand(N) - 80
    z = np.sin(4*np.pi*lat/180.0*np.pi) + np.cos(8*np.pi*lon/180.0*np.pi)

    return lat, lon, z


# Create Data
lat, lon, z = create2Ddata()

# Make bins
latbins = np.linspace(np.min(lat), np.max(lat), 75)
lonbins = np.linspace(np.min(lon), np.max(lon), 75)

# Bin the data
_, _, zi = make2dhist(lon, lat, z, latbins, lonbins)

fig = plt.figure()

# Just plot the scattered data
ax = plt.subplot(211, projection=PlateCarree())
plotmap()
plt.scatter(lon, lat, s=7, c=z, cmap='rainbow')

# Plot the binned 2D data
ax = plt.subplot(212, projection=PlateCarree())
plotmap()
plt.pcolormesh(
    lonbins, latbins, zi.T, shading='auto', transform=PlateCarree(),
    cmap='rainbow')
plt.show()


# %%

def create3Ddata():
    ''' Make random 3D data '''
    N = 8000
    lat = 10 * np.random.rand(N) + 40
    lon = 25 * np.random.rand(N) - 80
    t = 10 * np.random.rand(N)

    # Linearly changes sign of the cos+sin wavefield
    z = (t/5 - 1) * (np.sin(2*2*np.pi*lat/180.0*np.pi)
                     + np.cos(4*2*np.pi*lon/180.0*np.pi))

    return lat, lon, t, z


# Create Data
lat, lon, t, z = create3Ddata()

# Create bins
latbins = np.linspace(np.min(lat), np.max(lat), 75)
lonbins = np.linspace(np.min(lon), np.max(lon), 75)
tbins = np.linspace(np.min(t), np.max(t), 5)

# Bin the data
_, _, _, zi = make3dhist(lon, lat, t, z, latbins, lonbins, tbins)

# Normalize the colors so that variations in time are easily seen
norm = Normalize(vmin=-1.0, vmax=1.0)

fig = plt.figure(figsize=(12, 10))

# The scattered data in time bins
# Left column
for i in range(4):
    ax = plt.subplot(4, 3, 3*i + 1, projection=PlateCarree())
    plotmap()

    # Find points in time bins
    pos = np.where((tbins[i] < t) & (t < tbins[i+1]))

    # Plot scatter points
    plt.title(f'{tbins[i]:0.2f} < t < {tbins[i+1]:0.2f}')
    plt.scatter(lon[pos], lat[pos], c=z[pos], s=7, cmap='rainbow', norm=norm)
    plt.colorbar(orientation='horizontal', pad=0.0)

# Center column
for i in range(4):
    ax = plt.subplot(4, 3, 3*i + 2, projection=PlateCarree())
    plotmap()
    plt.title(f'{tbins[i]:0.2f} < t < {tbins[i+1]:0.2f}')

    # Find points in time bins
    pos = np.where((tbins[i] < t) & (t <= tbins[i+1]))

    # Bin the data
    _, _, zt = make2dhist(lon[pos], lat[pos], z[pos], latbins, lonbins)
    plt.pcolormesh(
        lonbins, latbins, zt.T, shading='auto', transform=PlateCarree(),
        cmap='rainbow', norm=norm)
    plt.colorbar(orientation='horizontal', pad=0.0)

for i in range(4):
    ax = plt.subplot(4, 3, 3*i + 3, projection=PlateCarree())
    plotmap()
    plt.title(f'{tbins[i]:0.2f} < t < {tbins[i+1]:0.2f}')
    plt.pcolormesh(
        lonbins, latbins, zi[:, :, i].T, shading='auto', transform=PlateCarree(),
        cmap='rainbow', norm=norm)
    plt.colorbar(orientation='horizontal', pad=0.0)


plt.show()
