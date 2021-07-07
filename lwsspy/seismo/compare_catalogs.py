import os
from typing import Optional
import lwsspy as lpy
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from obspy import Inventory
from cartopy.crs import PlateCarree, Mollweide
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm
from matplotlib.patches import Rectangle
from .cmt_catalog import CMTCatalog
from .source import CMTSource
from .plot_quakes import plot_quakes
from ..maps.plot_map import plot_map
from ..plot_util.plot_label import plot_label
from ..plot_util.remove_ticklabels import remove_topright


class CompareCatalogs:

    # Old and new parameters
    olat: np.ndarray
    nlat: np.ndarray
    olon: np.ndarray
    nlon: np.ndarray
    omoment: np.ndarray
    nmoment: np.ndarray
    odepth: np.ndarray
    ndepth: np.ndarray
    oeps_nu: np.ndarray
    neps_nu: np.ndarray

    # Map parameters
    central_longitude = 180.0

    # Plot style parameters
    cmt_cmap: ListedColormap = ListedColormap(
        [(0.9, 0.9, 0.9), (0.7, 0.7, 0.7), (0.5, 0.5, 0.5),
         (0.3, 0.3, 0.3), (0.1, 0.1, 0.1)])
    depth_dict = {0: (0.8, 0.2, 0.2), 70: (0.2, 0.6, 0.8), 300: (
        0.35, 0.35, 0.35), 800: (0.35, 0.35, 0.35)}
    depth_cmap: ListedColormap = ListedColormap(list(depth_dict.values()))
    depth_norm: Normalize = BoundaryNorm(list(depth_dict.keys()), depth_cmap.N)

    def __init__(self, old: CMTCatalog, new: CMTCatalog,
                 oldlabel: str = 'Old', newlabel: str = 'New',
                 stations: Inventory = None, nbins: int = 40):

        # Assign
        self.oldlabel = oldlabel
        self.newlabel = newlabel

        # Fix up so they can be compared (if both catalogs are very large this
        # can take some time)
        old = old.unique(ret=True)
        self.old, self.new = old.check_ids(new)
        self.N = len(self.old)

        # Number of bins
        self.nbins = nbins

        # Populate attributes
        self.populate()

        # Stations to be plotted alongside if defined
        self.stations = stations

    def populate(self):

        # Old and new values
        self.olat = self.old.getvals("latitude")
        self.nlat = self.new.getvals("latitude")
        self.olon = self.old.getvals("longitude")
        self.nlon = self.new.getvals("longitude")
        self.oM0 = self.old.getvals("M0")
        self.nM0 = self.new.getvals("M0")
        self.omoment_mag = self.old.getvals("moment_magnitude")
        self.nmoment_mag = self.new.getvals("moment_magnitude")
        self.odepth = self.old.getvals("depth_in_m")/1000.0
        self.ndepth = self.new.getvals("depth_in_m")/1000.0
        self.oeps_nu = self.old.getvals("decomp", "eps_nu")
        self.neps_nu = self.new.getvals("decomp", "eps_nu")
        self.otime_shift = self.old.getvals("time_shift")
        self.ntime_shift = self.new.getvals("time_shift")

        # Number of bins
        # Min max ddepth for cmt plotting
        self.ddepth = self.ndepth-self.odepth
        self.maxddepth = np.max(self.ddepth)
        self.minddepth = np.min(self.ddepth)
        self.dd_absmax = np.max(np.abs(
            [np.quantile(np.min(self.ddepth), 0.30),
             np.quantile(np.min(self.ddepth), 0.70)]))
        self.maxdepth = np.max(self.odepth)
        self.mindepth = np.min(self.odepth)
        self.dmbins = np.linspace(-0.5, 0.5 + 0.5 / self.nbins, self.nbins)
        self.ddegbins = np.linspace(-0.1, 0.1 + 0.1 / self.nbins, self.nbins)
        self.dzbins = np.linspace(-self.dd_absmax,
                                  2 * self.dd_absmax / self.nbins, self.nbins)
        self.dtbins = np.linspace(-10, 10 + 10 / self.nbins, self.nbins)

    def plot_cmts(self):

        # Get axes (must be map axes)
        ax = plt.gca()

        # Plot events
        scatter, ax, l1, l2 = plot_quakes(
            self.nlat, self.nlon, self.ndepth, self.nmoment_mag, ax=ax,
            yoffsetlegend2=0.09, sizefunc=lambda x: (x-(np.min(x)-1))**2.5 + 5)
        ax.set_global()
        plot_map(zorder=0, fill=True)
        plot_label(ax, f"N: {self.N}", location=1, box=False, dist=0.0)

    def plot_eps_nu(self):

        # Get axes
        ax = plt.gca()

        bins = np.arange(-0.5, 0.50001, 0.01)

        # Plot histogram GCMT3D
        plt.hist(self.oeps_nu[:, 0], bins=bins, edgecolor='k',
                 facecolor=(0.3, 0.3, 0.8, 0.75), linewidth=0.75,
                 label='GCMT', histtype='stepfilled')

        # Plot histogram GCMT3D+
        plt.hist(self.neps_nu[:, 0], bins=bins, edgecolor='k',
                 facecolor=(0.3, 0.8, 0.3, 0.75), linewidth=0.75,
                 label='GCMT3D+', histtype='stepfilled')
        plt.legend(loc='upper left', frameon=False, fancybox=False,
                   numpoints=1, scatterpoints=1, fontsize='x-small',
                   borderaxespad=0.0, borderpad=0.5, handletextpad=0.2,
                   labelspacing=0.2, handlelength=1.0,
                   bbox_to_anchor=(0.0, 1.0))

        # Plot stats label
        label = (
            f"{self.oldlabel}\n"
            f"$\\mu$ = {np.mean(self.oeps_nu[:,0]):7.4f}\n"
            f"$\\sigma$ = {np.std(self.oeps_nu[:,0]):7.4f}\n"
            f"{self.newlabel}\n"
            f"$\\mu$ = {np.mean(self.neps_nu[:,0]):7.4f}\n"
            f"$\\sigma$ = {np.std(self.neps_nu[:,0]):7.4f}\n")
        plot_label(ax, label, location=2, box=False,
                   fontdict=dict(fontsize='xx-small', fontfamily="monospace"))
        plot_label(ax, "CLVD-", location=6, box=False,
                   fontdict=dict(fontsize='small'))
        plot_label(ax, "CLVD+", location=7, box=False,
                   fontdict=dict(fontsize='small'))
        plot_label(ax, "DC", location=14, box=False,
                   fontdict=dict(fontsize='small'))
        plt.xlabel(r"$\epsilon$")

    def plot_depth_v_ddepth(self):

        # Get axis
        ax = plt.gca()
        msize = 15

        plt.scatter(
            self.ddepth, self.odepth,
            c=self.depth_cmap(self.depth_norm(self.odepth)),
            s=msize, marker='o', alpha=0.5, edgecolors='none')

        # Custom legend
        classes = ['  <70 km', ' ', '>300 km']
        colors = [(0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.35, 0.35, 0.35)]
        for cla, col in zip(classes, colors):
            plt.scatter([], [], c=[col], s=msize, label=cla, alpha=0.5,
                        edgecolors='none')
        plt.legend(loc='lower left', frameon=False, fancybox=False,
                   numpoints=1, scatterpoints=1, fontsize='x-small',
                   borderaxespad=0.0, borderpad=0.5, handletextpad=0.2,
                   labelspacing=0.2, handlelength=0.5,
                   bbox_to_anchor=(0.0, 0.0))

        # Zero line
        plt.plot([0, 0], [0, np.max(self.odepth)],
                 "k--", lw=1.5)

        # Axes properties
        plt.ylim(([10, np.max(self.odepth)]))
        plt.xlim(([np.min(self.ddepth), np.max(self.ddepth)]))
        ax.invert_yaxis()
        ax.set_yscale('log')
        plt.xlabel("Depth Change [km]")
        plt.ylabel("Depth [km]")

    def plot_slab_map(self, outfile: Optional[str] = None, extent=None):

        if outfile is not None:
            backend = plt.get_backend()
            plt.switch_backend('pdf')

        # Get slab location
        dss = lpy.get_slabs()
        vmin, vmax = lpy.get_slab_minmax(dss)

        # Compute levels
        levels = np.linspace(vmin, vmax, 100)

        # Define color mapping
        cmap = plt.get_cmap('rainbow')
        norm = BoundaryNorm(levels, cmap.N)

        # Get extent in which to include surrounding slabs
        lats = self.olat
        lons = self.olon

        # Plot map
        proj = Mollweide(central_longitude=160.0)
        ax = plt.axes(projection=proj)
        if extent is not None:
            ax.set_extent(extent)
        else:
            ax.set_global()
        lpy.plot_map()
        lpy.plot_map(fill=False, borders=False, zorder=10)
        # ax.set_extent(extent05)

        # Plot map with central longitude on event longitude
        lpy.plot_slabs(dss=dss, levels=levels, cmap=cmap, norm=norm)

        # Plot CMT as popint or beachball
        # cdepth = cmap(norm(-1*self.odepth))

        # Old CMTs
        plt.plot(
            np.vstack((self.olon, self.nlon)),
            np.vstack((self.olat, self.nlat)), 'k',
            linewidth=0.75, transform=PlateCarree(), zorder=105)

        plt.scatter(lons, lats, c=-1*self.odepth, s=30,
                    marker='s', edgecolor='k', cmap=cmap, norm=norm,
                    transform=PlateCarree(), zorder=100)

        # New CMTs
        plt.scatter(self.nlon, self.nlat, c=-1*self.odepth, s=30,
                    marker='o', edgecolor='k', cmap=cmap, norm=norm,
                    transform=PlateCarree(), zorder=110)

        # Redo, I think?
        if extent is not None:
            ax.set_extent(extent)

        c = lpy.nice_colorbar(aspect=40, fraction=0.1, shrink=0.6)
        c.set_label("Depth [km]")

        if outfile is not None:
            plt.savefig(outfile, dpi=300)
            plt.switch_backend(backend)
        else:
            plt.show()

    def plot_summary(self, outfile: Optional[str] = None):

        if outfile is not None:
            backend = plt.get_backend()
            plt.switch_backend('pdf')

        # Create figure handle
        fig = plt.figure(figsize=(11, 6))

        # Create subplot layout
        GS = GridSpec(3, 3)
        plt.subplots_adjust(wspace=0.25, hspace=0.75)
        # Plot events
        ax = fig.add_subplot(GS[:2, :2])
        ax.axis('off')
        pad, w, h = 0.025, 0.95, 0.825
        a = ax.get_position()
        iax_pos = [a.x1-(w+pad)*a.width, a.y1-(h+pad) *
                   a.height, w*a.width, h*a.height]
        iax = fig.add_axes(iax_pos, projection=Mollweide(
            central_longitude=self.central_longitude))
        iax.set_global()
        self.plot_cmts()
        plot_label(ax, 'a)', location=6, box=False)

        # Plot eps_nu change
        ax = fig.add_subplot(GS[0, 2])
        self.plot_eps_nu()
        plot_label(ax, 'b)', location=17, box=False)

        # Plot Depth v dDepth
        ax = fig.add_subplot(GS[1, 2])
        self.plot_depth_v_ddepth()
        plot_label(ax, 'c)', location=6, box=False)

        # Plot tshift histogram
        tbins = self.nbins
        # tbins = np.linspace(-0.5, 0.5, 100)
        ax = fig.add_subplot(GS[2, 0])
        self.plot_histogram(
            self.ntime_shift-self.otime_shift, tbins,
            facecolor='lightgray')
        remove_topright()
        plt.xlabel("Centroid Time Change [sec]")
        plt.ylabel("N", rotation=0, horizontalalignment='right')
        plot_label(ax, 'd)', location=6, box=False)

        # Plot Scalar Moment histogram
        Mbins = self.nbins
        # Mbins = np.linspace(-10, 10, 100)
        ax = fig.add_subplot(GS[2, 1])
        self.plot_histogram(
            (self.nM0-self.oM0)/self.oM0*100, Mbins,
            facecolor='lightgray', statsleft=False)
        remove_topright()
        plt.xlabel("Scalar Moment Change [%]")
        plt.ylabel("N", rotation=0, horizontalalignment='right')
        plot_label(ax, 'e)', location=6, box=False)

        # Plot ddepth histogram
        zbins = self.nbins
        # zbins = np.linspace(-2.5, 2.5, 100)
        ax = fig.add_subplot(GS[2, 2])
        self.plot_histogram(
            self.ndepth-self.odepth, zbins, facecolor='lightgray',
            statsleft=True)
        remove_topright()
        plt.xlabel("Depth Change [km]")
        plt.ylabel("N", rotation=0, horizontalalignment='right')
        plot_label(ax, 'f)', location=6, box=False)

        if outfile is not None:
            plt.savefig(outfile)
            plt.switch_backend(backend)

    @staticmethod
    def get_change(o, n, d: bool, f: bool):
        """Getting the change values, if ``d`` is ``False`` the function just
        returns (o, n).

        Parameters
        ----------
        o : arraylike
            old values
        n : arraylike
            new values
        d : bool
            compute change
        f : bool
            compute fractional change

        Returns
        -------
        tuple
            old, new values
        """

        if d:
            if f:
                o_out = (n - o)/o
            else:
                o_out = (n - o)
            n_out = o_out

        else:
            o_out = o
            n_out = n
        return o_out, n_out

    def plot_2D_scatter(
            self, param1="depth", param2="M0",
            d1: bool = True,
            d2: bool = True,
            f1: bool = False,
            f2: bool = False,
            xlog: bool = False,
            ylog: bool = False,
            xrange: Optional[list] = None,
            yrange: Optional[list] = None,
            xinvert: bool = False,
            yinvert: bool = False,
            nbins: int = 40,
            outfile: Optional[str] = None):
        """
        latitude = lat
        longitude = lon
        depth = depth
        M0 = M0
        moment_magnitude = moment_mag
        eps_nu = eps_nu

        d? meaning the change in the parameter, boolean

        f? meaning fractional change, boolean, only used of d? True,
        """

        if outfile is not None:
            backend = plt.get_backend()
            plt.switch_backend('pdf')
            plt.figure(figsize=(4.5, 3))

        # Get first parameters
        old1 = getattr(self, "o" + param1)
        new1 = getattr(self, "n" + param1)

        # Get second parameters
        old2 = getattr(self, "o" + param2)
        new2 = getattr(self, "n" + param2)

        # Compute the values to be plotted
        o1p, n1p = self.get_change(old1, new1, d1, f1)
        o2p, n2p = self.get_change(old2, new2, d2, f2)

        # Plot 2D scatter histograms
        if d1 and d2:
            label = " "
            axscatter, _, _ = lpy.scatter_hist(
                n1p,
                n2p,
                nbins,
                label=label,
                histc=(0.4, 0.4, 1.0),
                fraction=0.85,
                mult=False)

        elif (d1 and not d2) or (not d1 and d2):
            label = " "
            axscatter, _, _ = lpy.scatter_hist(
                n1p,
                n2p,
                nbins,
                label=label,
                histc=(0.4, 0.4, 1.0),
                fraction=0.85,
                mult=False)

        else:
            labels = ["O", "N"]
            axscatter, _, _ = lpy.scatter_hist(
                [o1p, n1p],
                [o2p, n2p],
                nbins,
                label=labels,
                histc=[(0.3, 0.3, 0.9), (0.9, 0.3, 0.3)],
                fraction=0.85,
                mult=True)

        # Possibly do stuff with axes
        if d1 and not d2:
            xlabel = "d" + param1.capitalize()
            ylabel = param2.capitalize()
        elif not d1 and d2:
            xlabel = param1.capitalize()
            ylabel = "d" + param2.capitalize()
        elif not d1 and not d2:
            xlabel = param1.capitalize()
            ylabel = param2.capitalize()
        else:
            xlabel = "d" + param1.capitalize()
            ylabel = "d" + param2.capitalize()

        # add labels to plot
        axscatter.set_xlabel(xlabel)
        axscatter.set_ylabel(ylabel)

        if ylog:
            axscatter.set_yscale('log')
        if xlog:
            axscatter.set_xscale('log')

        if xrange is not None:
            axscatter.set_xlim(xrange)
        if yrange is not None:
            axscatter.set_ylim(yrange)

        if xinvert:
            axscatter.invert_xaxis()
        if yinvert:
            axscatter.invert_yaxis()

        if outfile is not None:
            plt.savefig(outfile)
            plt.switch_backend(backend)
            plt.close()

    def plot_depth_v_eps_nu(self, outfile=None):

        if outfile is not None:
            backend = plt.get_backend()
            plt.switch_backend('pdf')
            plt.figure(figsize=(4.5, 3))

        axscatter, _, _ = lpy.scatter_hist(
            [self.oeps_nu[:, 0], self.neps_nu[:, 0]],
            [self.odepth, self.ndepth],
            self.nbins,
            label=[self.oldlabel, self.newlabel],
            histc=[(0.4, 0.4, 1.0), (1.0, 0.4, 0.4)],
            fraction=0.85, ylog=True,
            mult=True)
        axscatter.invert_yaxis()
        axscatter.set_xlim((-0.5, 0.5))
        ylim = axscatter.get_ylim()
        axscatter.set_ylim(
            (ylim[0], 0.95*np.min((np.min(self.ndepth), np.min(self.odepth)))))
        axscatter.set_yscale('log')

        # Plot clvd labels
        plot_label(axscatter, "CLVD-", location=11, box=False,
                   fontdict=dict(fontsize='small'))
        plot_label(axscatter, "CLVD+", location=10, box=False,
                   fontdict=dict(fontsize='small'))
        plot_label(axscatter, "DC", location=16, box=False,
                   fontdict=dict(fontsize='small'))
        axscatter.tick_params(labelbottom=False)
        plt.ylabel('Depth [km]')

        if outfile is not None:
            plt.savefig(outfile)
            plt.switch_backend(backend)
            plt.close()

    def plot_histogram(self, ddata, n_bins, facecolor=(0.7, 0.2, 0.2),
                       alpha=1, chi=False, wmin=None, statsleft: bool = False,
                       label: str = None, stats: bool = True,
                       CI: bool = False):
        """Plots histogram of input data."""

        if wmin is not None:
            print(f"Datamin: {np.min(ddata)}")
            ddata = ddata[np.where(ddata >= wmin)]
            print(f"Datamin: {np.min(ddata)}")

        # The histogram of the data
        ax = plt.gca()
        n, bins, _ = ax.hist(ddata, n_bins, facecolor=facecolor,
                             edgecolor=facecolor, alpha=alpha, label=label)
        _, _, _ = ax.hist(ddata, n_bins, color='k', histtype='step')
        text_dict = {
            "fontsize": 'x-small',
            "verticalalignment": 'top',
            "horizontalalignment": 'right',
            "transform": ax.transAxes,
            "zorder": 100,
            'family': 'monospace'
        }

        if stats:
            # Get datastats
            datamean = np.mean(ddata)
            datastd = np.std(ddata)

            # Check if mean closer to right edge or left edge and putt stats
            # wherever there is more room
            xmin, xmax = ax.get_xlim()
            if np.abs(datamean - xmin) > np.abs(xmax - datamean):
                statsleft = True
            else:
                statsleft = False

            if statsleft:
                text_dict["horizontalalignment"] = 'left'
                posx = 0.03
            else:
                posx = 0.97

            ax.text(posx, 0.97,
                    f"$\\mu$ = {datamean:5.2f}\n"
                    f"$\\sigma$ = {datastd:5.2f}",
                    **text_dict)

        if CI:
            ci_norm = {
                "80": 1.282,
                "85": 1.440,
                "90": 1.645,
                "95": 1.960,
                "99": 2.576,
                "99.5": 2.807,
                "99.9": 3.291
            }
            if chi:
                Zval = ci_norm["90"]
            else:
                Zval = ci_norm["95"]

            mean = np.mean(ddata)
            pmfact = Zval * np.std(ddata)
            nCI = [mean - pmfact, mean + pmfact]
            # if we are only concerned about the lowest values the more
            # the better:
            if wmin is not None:
                nCI[1] = np.max(ddata)
                if nCI[0] < wmin:
                    nCI[0] = wmin
            minbox = [np.min(bins), 0]
            minwidth = (nCI[0]) - minbox[0]
            maxbox = [nCI[1], 0]
            maxwidth = np.max(bins) - maxbox[0]
            height = np.max(n)*1.05

            boxdict = {
                "facecolor": 'w',
                "edgecolor": None,
                "alpha": 0.6,
            }
            minR = Rectangle(minbox, minwidth, height, **boxdict)
            maxR = Rectangle(maxbox, maxwidth, height, **boxdict)
            ax.add_patch(minR)
            ax.add_patch(maxR)

            return nCI
        else:
            return None

    def filter(self, maxdict: dict = dict(), mindict: dict = dict()):
        """This uses two dictionaries as inputs. One dictionary for
        maximum values and one dictionary that contains min values of the
        elements to filter. To do that we create a dictionary containing
        the attributes and properties of
        :class:``lwsspy.seismo.source.CMTSource``.

        List of Attributes and Properties
        -------------------------

        .. literal::

            origin_time
            pde_latitude
            pde_longitude
            pde_depth_in_m
            mb
            ms
            region_tag
            eventname
            cmt_time
            half_duration
            latitude
            longitude
            depth_in_m
            m_rr
            m_tt
            m_pp
            m_rt
            m_rp
            m_tp
            M0
            moment_magnitude
            time_shift

        Example
        -------

        Let's filter the catalog to only contain events with a maximum depth
        of 20km.

        >>> maxfilterdict = dict(depth_in_m=20000.0)
        >>> cmtcat = CMTCatalog.from_files("CMTfiles/*")
        >>> filtered_cat = cmtcat.filter(maxdict=maxfilterdict)

        will returns a catalog with events shallower than 20.0 km.
        """

        # Create new list of cmts
        oldlist, newlist = deepcopy(self.old.cmts), deepcopy(self.new.cmts)

        # percentage parameters
        pparams = [
            "m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp", "M0",
            "moment_magnitude"]

        # First maxvalues
        for key, value in maxdict.items():

            # Create empty pop set
            popset = set()
            print(key, value)
            # Check CMTs that are below threshold for key
            for _i, (_ocmt, _ncmt) in enumerate(zip(oldlist, newlist)):
                oldval = getattr(_ocmt, key)
                newval = getattr(_ncmt, key)
                if key in pparams:
                    if np.abs((newval-oldval)/oldval) > value:
                        popset.add(_i)
                else:
                    if np.abs(newval-oldval) > value:
                        popset.add(_i)

            # Convert set to list and sort
            poplist = list(popset)
            poplist.sort()

            # Pop found indeces
            for _i in poplist[::-1]:
                oldlist.pop(_i)
                newlist.pop(_i)

        # First maxvalues
        for key, value in mindict.items():

            # Create empty pop set
            popset = set()

            # Check CMTs that are below threshold for key
            for _i, (_ocmt, _ncmt) in enumerate(zip(oldlist, newlist)):
                oldval = getattr(_ocmt, key)
                newval = getattr(_ncmt, key)
                if key in pparams:
                    if np.abs((newval-oldval)/oldval) < value:
                        popset.add(_i)
                else:
                    if np.abs(newval-oldval) < value:
                        popset.add(_i)

            # Convert set to list and sort
            poplist = list(popset)
            poplist.sort()

            # Pop found indeces
            for _i in poplist[::-1]:
                oldlist.pop(_i)
                newlist.pop(_i)

        return CompareCatalogs(CMTCatalog(oldlist), CMTCatalog(newlist),
                               oldlabel=self.oldlabel, newlabel=self.newlabel,
                               nbins=self.nbins)


def bin():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--old', dest='old',
                        help='Old cmt solutions', nargs='+',
                        required=True, type=str)
    parser.add_argument('-n', '--new', dest='new',
                        help='New cmt solutions', nargs='+',
                        required=True, type=str)
    parser.add_argument('-d', '--outdir', dest='outdir',
                        help='Directory to place outputs in',
                        type=str, default='.')
    parser.add_argument('-w', '--write-cats', dest='write',
                        help='Write catalogs to file', action='store_true',
                        default=False)
    parser.add_argument('-ol', '--old-label', dest='oldlabel',
                        help='Old label',
                        required=True, type=str or None)
    parser.add_argument('-nl', '--new-label', dest='newlabel',
                        help='New label',
                        required=True, type=str or None)
    args = parser.parse_args()

    # Get catalogs
    old = lpy.CMTCatalog.from_file_list(args.old)
    new = lpy.CMTCatalog.from_file_list(args.new)

    print("Old:", len(old.cmts))
    print("New:", len(new.cmts))

    # Get overlaps
    ocat, ncat = old.check_ids(new)

    print("After checkid:")
    print("  Old:", len(ocat.cmts))
    print("  New:", len(ncat.cmts))

    # Writing
    if args.write:
        ocat.save(os.path.join(args.outdir, args.oldlabel + ".pkl"))
        ncat.save(os.path.join(args.outdir, args.newlabel + ".pkl"))

    # Compare Catalog
    CC = lpy.CompareCatalogs(old=ocat, new=ncat,
                             oldlabel=args.oldlabel, newlabel=args.newlabel,
                             nbins=25)

    # CC.plot_2D_scatter(param1="moment_mag", param2="depth", d1=False,
    #                    d2=False, xlog=False, ylog=True, yrange=[3, 800],
    #                    yinvert=True)
    # CC.plot_2D_scatter(param1="depth", param2="depth", d1=True,
    #                    d2=False, xlog=False, ylog=False, yrange=[0, 700],
    #                    yinvert=True)
    # CC.plot_2D_scatter(param1="depth", param2="time_shift", d1=True,
    #                    d2=True)
    # CC.plot_2D_scatter(param1="time_shift", param2="depth", d1=True,
    #                    d2=False, ylog=False, yrange=[0, 700],
    #                    yinvert=True)

    # CC.plot_depth_v_eps_nu()
    # plt.show(block=True)

    # Filter for a minimum depth larger than zero
    # CC = CC.filter(mindict={"depth_in_m": 10000.0})
    # for ocmt, ncmt in zip(CC.old, CC.new):
    #     print(f"\n\nOLD: {(ncmt.depth_in_m - ocmt.depth_in_m)/1000.0}")
    #     print(ocmt)
    #     print(" ")
    #     print("NEW")
    #     print(ncmt)

    # extent = -80, -60, -10, -30
    # extent = None

    # Comparison figures
    # CC.plot_slab_map()
    # outfile=os.path.join(
    #     args.outdir, "catalog_slab_map.pdf"), extent=extent)
    CC.plot_summary(outfile=os.path.join(
        args.outdir, "catalog_comparison.pdf"))
    CC.plot_depth_v_eps_nu(outfile=os.path.join(
        args.outdir, "depth_v_sourcetype.pdf"))
