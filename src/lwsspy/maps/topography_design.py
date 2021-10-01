import os
import matplotlib.pyplot as plt
from osgeo import gdal, osr
import cartopy.crs as ccrs
import numpy as np
import lwsspy as lpy
from matplotlib.colors import Normalize
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter
from scipy import ndimage, misc


def start_stop(a, trigger_val):
    """Very simple linear edge detection"""
    # "Enclose" mask with sentients to catch shifts later on
    mask = np.r_[False, np.equal(a, trigger_val), False]

    # Get the shifting indices
    idx = np.flatnonzero(mask[1:] != mask[:-1])

    # Get the start and end indices with slicing along the shifting ones
    return zip(idx[::2], idx[1::2]-1)


class TopographyDesign:

    mdata: np.ndarray
    taper: np.ndarray

    def __init__(self, lat: np.array, lon: np.array, data: np.ndarray,
                 offset: int = 30, exag: float = 75, cmap='rainbow',
                 facecolor="w", mask_val: float = None,
                 filter_val: float = 0.0025):

        # Init
        self.lat = lat
        self.lon = lon
        self.dimlat = len(self.lat)
        self.dimlon = len(self.lon)
        self.data = data
        self.offset = offset
        self.dlat = self.lat[1] - self.lat[0]
        self.cmap = cmap
        self.mask_val = mask_val
        self.facecolor = facecolor
        self.filter_val = filter_val
        self.exag = exag

        # Get the data of the data
        self.get_data_masks()

        # Pre data for plotting
        self.data_prep()

    def get_data_masks(self):

        if self.mask_val is not None:
            self.mdata = np.where(
                self.data == self.mask_val, np.nan, self.data)
            self.taper = np.where(self.data == self.mask_val, 0, 1)

            self.mindata = np.min(self.data[self.data != self.mask_val])
            self.maxdata = np.max(self.data[self.data != self.mask_val])
        else:
            self.mdata = self.data
            self.mindata = np.min(self.data)
            self.maxdata = np.max(self.data)

    def data_prep(self):

        # Defined norm to control the topography
        self.normalize()

        # If filter filter if no, don't
        self.filter()

    def filter(self):

        if self.filter_val is not None:
            # Filter
            self.fdata = gaussian_filter(
                self.normdata, self.filter_val * (self.dimlon + self.dimlat)/2)

    def normalize(self):

        # Create Norm
        self.norm = Normalize(vmin=self.mindata, vmax=self.maxdata)

        # Normalize the topography for plotting the line
        self.normdata = np.where(np.isnan(self.mdata),
                                 0.0, self.norm(self.mdata))

        # Actual data needed for the plotting of the line colors
        self.plotdata = np.where(np.isnan(self.mdata), 0.0, self.mdata)

    def _plot(self, k=0, figsize=(16, 9)):

        print(f"Computing Angle {k}...")

        # Rotation
        if k == 0:
            r_normdata = self.fdata
            r_plotdata = self.plotdata
            if self.mask_val is not None:
                r_taper = self.taper

        else:
            r_normdata = ndimage.rotate(self.fdata, - k, reshape=False)
            r_plotdata = ndimage.rotate(self.plotdata, - k, reshape=False)
            if self.mask_val is not None:
                r_taper = ndimage.rotate(self.taper, - k, reshape=False)

        # Create figure
        fig = plt.figure(facecolor=self.facecolor, figsize=figsize)
        ax = plt.axes(facecolor=self.facecolor)
        ax.axis('off')

        for _i, _ilat in enumerate(self.lat[::self.offset]):

            # Get the masks
            if self.mask_val is not None:

                # Edge detection
                masks = start_stop(r_taper[_i * self.offset, :], trigger_val=1)

                # Empty taper
                ltaper = np.zeros_like(self.lon)

                # Create series of taper windows
                for start, stop in masks:
                    ltaper[start:stop] = tukey(stop - start, alpha=0.25)

            else:

                # Taper for the edges in case no maskval is given
                ltaper = tukey(self.dimlon, alpha=0.25)

            # Just the latitude line
            yt = _ilat * np.ones_like(self.lon)

            # Add exagerated topography values
            offexag = self.dlat * self.offset * self.exag
            yt += (r_normdata[_i * self.offset, :] * offexag)

            # The actual topotgrah
            z = r_plotdata[_i * self.offset, :]

            # Linewidth decreasing with taper --> edges fade
            linewidths = r_normdata[_i * self.offset, :] * ltaper

            # Baseline for the plot_between function so the lower elevations
            # in the back are not visible
            baseline = np.min(self.dlat) - (self.dlat * self.exag)

            # Plot polygons to lines in the back aren't visible
            plt.fill_between(
                self.lon, yt, y2=baseline,
                zorder=-_i, facecolor=self.facecolor, edgecolor='none'
            )

            # Plot lines with linewidth and topo cmap
            lines, sm = lpy.plot_xyz_line(
                self.lon, yt, z, cmap=self.cmap, norm=self.norm,
                capstyle='round', linewidths=linewidths, zorder=-_i-0.5,
                clip_on=False
            )

        # Set Axes limits
        ax.set_xlim(np.min(self.lon), np.max(self.lon))
        ax.set_ylim(
            np.min(self.lat),
            np.max(self.lat) + self.dlat * self.offset * self.exag
        )

        # Adjust boundaries of the plot
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    def plot(self, k=0, outfile: str = None, **kwargs):

        # Plot
        self._plot(k=k, **kwargs)

        # Save or show
        if outfile is not None:
            plt.savefig(outfile, dpi=300)
            plt.close()
        else:
            plt.show()

    def animate(self, step=90, animationdir: str = "animation", **kwargs):

        # Create dir if doesnt exist
        if os.path.exists(animationdir) is False:
            os.makedirs(animationdir)

        for _k in range(0, 360, step):

            self._plot(k=_k, **kwargs)

            # Plot label
            label = f'N{str(np.abs(_k)).zfill(3)}'  # NXXX deg aximuth
            color = 'w' if self.facecolor == 'k' else 'k'
            lpy.plot_label(
                plt.gca(), label, location=1, color=color,
                box=False, fontdict=dict(fontsize="x-large")
            )

            if animationdir is not None:
                plt.savefig(
                    os.path.join(
                        animationdir,
                        f"azimuth_{str(np.abs(_k)).zfill(3)}.png"), dpi=300)
                plt.close()
