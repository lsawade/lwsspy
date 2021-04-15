import os
import lwsspy as lpy
import numpy as np
from numpy.typing import ArrayLike
import re
from matplotlib.dates import datestr2num
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t
from matplotlib.backends.backend_pdf import PdfPages


def naninfmin(a):
    return np.nanmin(a[np.abs(a) != np.inf])


def naninfmax(a):
    return np.nanmax(a[np.abs(a) != np.inf])


class River:
    name: str
    site: str
    years: ArrayLike
    discharge: ArrayLike
    stage: ArrayLike

    def __init__(self, stationnumber: str, flood_stage: float = None,
                 datadir: str = os.path.join(lpy.DOCFIGURESCRIPTDATA),
                 figuredir: str = os.path.join(lpy.DOCFIGURES),
                 pre_title: str = None,
                 save: bool = False):

        self.site = stationnumber.zfill(8)
        self.flood_stage = flood_stage
        self.url = f'https://nwis.waterdata.usgs.gov/nwis/peak?site_no={self.site}&agency_cd=USGS&format=rdb'
        self.datadir = datadir
        self.figuredir = figuredir
        self.save = save
        self.pre_title = pre_title

    def populate(self):

        # River file
        riverdir = os.path.join(self.datadir, "rivers")
        if os.path.exists(riverdir) is False:
            os.makedirs(riverdir)
        riverfile = os.path.join(
            riverdir, f"river_{self.site}.txt")

        # Download if I don't have it
        if not os.path.exists(riverfile):
            lpy.downloadfile(self.url, riverfile)

        # Load file as string
        with open(riverfile, 'r') as f:
            riverstr = f.read()
            if 'No sites/data found using the selection criteria specified' in riverstr:
                raise ValueError(
                    "Double Check your site number, no data for the current one")

            self.name = self.get_station_name(riverstr)

        _, years, discharge, stage = self.get_station_table(
            riverfile)

        # Fix gaps
        def fix_gaps(x, pos):
            x = x.astype(float)
            return np.hstack(
                (x[0], np.insert(x[1:], pos, np.nan)))

        pos = np.where(np.diff(years) > 1.0)[0]
        if len(pos) > 0:
            self.years = fix_gaps(years, pos)
            self.discharge = fix_gaps(discharge, pos)
            self.stage = fix_gaps(stage, pos)
        else:
            self.years = years
            self.discharge = discharge.astype(float)
            self.stage = stage.astype(float)

    @staticmethod
    def get_station_name(riverstr: str):
        return ' '.join(riverstr.split('\n')[33].split()[1:])

    @staticmethod
    def get_station_table(riverfile):
        """Takes in the location of the tab-separated river data file and outputs
        tuple of arrays containing the.
        (siteno, year, discharge, stage)

        Parameters
        ----------
        riverfile : str
            TAB-sparated USGS River discharge file

        Returns
        -------
        tuple
            (siteno, year, discharge, stage)
        """

        # Read file
        # Columns are
        # agency_cd	site_no	peak_dt	peak_tm	peak_va	peak_cd	gage_ht	gage_ht_cd
        # Comments removes the header underneath the comments
        comments = ["#", 'agency', '5s']
        strarray = np.loadtxt(
            riverfile, dtype=str, comments=comments,
            usecols=(0, 1, 2, 3, 4, 5, 6, 7), delimiter='\t')

        # Convert values to
        def str2num(x: str):
            x = x.split('-')[0]
            return int(x[:4])

        site = strarray[0, 1]
        date = np.array([str2num(x) for x in strarray[:, 2]])
        discharge = np.array(
            [float(x) if x != '' else np.nan for x in strarray[:, 4]])
        stage = np.array(
            [float(x) if x != '' else np.nan for x in strarray[:, 6]])

        return site, date, discharge, stage

    @staticmethod
    def plot_peak_annual_discharge(r, flood_stage=None):
        ax = plt.gca()
        plt.plot(r.years, r.discharge, ".-k")
        plt.xlabel(r'Year')
        plt.ylabel(r'Discharge [f$^3$/s]')
        plt.grid('on')

        # Add flood stage line
        if flood_stage is not None:
            # Compute relationship
            istage = np.argsort(r.stage)
            T = lpy.Trendline(r.stage[istage], r.discharge[istage])
            T.fit()

            # Plot floodstage line
            plt.plot(
                (np.min(r.years), np.max(r.years)),
                (T.model(flood_stage), T.model(flood_stage)),
                label="Flood Stage")
            plt.legend()

        # Set limits
        ax.set_xlim((naninfmin(r.years), naninfmax(r.years)))
        ax.set_ylim((0, 1.25*naninfmax(r.discharge)))

    @staticmethod
    def plot_stage_v_discharge(r):

        # Stage Trendline
        istage = np.argsort(r.stage)
        T = lpy.Trendline(
            r.stage[istage], r.discharge[istage], pred_type='2')
        T.fit()
        x_trend = np.linspace(naninfmin(r.stage),  naninfmax(r.stage), 1000)

        ax = plt.gca()
        plt.plot(r.stage, r.discharge, ".k")
        T.plot_trendline(x_trend, 'k', alpha=0.75)
        T.plot_confidence_band(
            x_trend, ec='none', fc='lightgrey', zorder=0)
        plt.xlabel(r'Stage [f]')
        plt.ylabel(r'Discharge [f$^3$/s]')
        plt.grid('on')
        lpy.plot_label(ax, fr'$R^2 = {T.R2:4.2f}$',
                       location=1, fontdict=dict(fontsize='small'))

        # Set limits
        ax.set_xlim((naninfmin(r.stage),  naninfmax(r.stage)))
        ax.set_ylim((0, 1.25*naninfmax(r.discharge)))

    def plot_RI_v_discharge(self):

        # Get only values that don't have a nan
        nonnanpos = ~np.isnan(self.years) & ~np.isnan(self.discharge)
        nonnandischarge = self.discharge[nonnanpos]
        nonnanyears = self.years[nonnanpos]
        N = len(nonnandischarge)
        preRI = np.arange(N+1, 1, -1)

        # Sort values after discharge
        idischarge = np.argsort(nonnandischarge)

        # Sort values after year
        iyear = np.argsort(nonnanyears)
        iyeardischarge = np.argsort(iyear[idischarge])

        # Compute RI
        self.RI = ((N+1)/preRI[iyeardischarge])
        self.nonnandischarge = nonnandischarge
        iRI = np.argsort(self.RI)

        # Compute Trendline
        T = lpy.Trendline(self.RI[iRI], nonnandischarge[iRI], pred_type='log')
        T.fit()
        discharge_predict_100 = T.model(100.0)

        # Plot RI and its 100yr prediction
        ax = plt.gca()
        plt.plot(self.RI, nonnandischarge, ".k")
        T.plot_trendline(np.linspace(1, 100, 1000), "k", alpha=0.75)
        T.plot_confidence_band(np.linspace(1, 100, 1000),
                               ec='none', fc='lightgrey', zorder=0)

        # Plot identifier
        plt.plot((100, 100), (0, 1.25*discharge_predict_100),
                 'k', linewidth=0.75)
        plt.plot((0, 125), (discharge_predict_100, discharge_predict_100),
                 'k', linewidth=0.75)
        plt.plot(100, discharge_predict_100, 'ok',
                 markersize=10.0, markerfacecolor='none')

        # Plot labels
        plt.xlabel(r'RI [year]')
        plt.ylabel(r'Discharge [f$^3$/s]')
        lpy.plot_label(
            ax,
            f'$R^2 = {T.R2:4.2f}$\n'
            f'100yr = {int(discharge_predict_100)}',
            location=1, fontdict=dict(fontsize='small'))

        # Change axes style
        plt.grid('on')
        ax.set_xscale('log')

        # Change limits
        ax.set_xlim((1, 125))
        # ax.set_ylim((0, 1.25*discharge_predict_100))

    def plot_summary(self):

        fig = plt.figure(figsize=(12, 10))
        plt.subplots_adjust(wspace=0.1)

        if self.pre_title is None:
            plt.suptitle(self.name)
        else:
            plt.suptitle(fr"$\bf{{{self.pre_title}}}$: " + self.name)

        plt.subplot(221)
        plt.title("Peak Annual Discharge")
        self.plot_peak_annual_discharge(self, flood_stage=self.flood_stage)

        ax = plt.subplot(222)
        plt.title("Stage v Discharge Relationship")
        self.plot_stage_v_discharge(self)
        ax.tick_params(labelleft=False)
        plt.ylabel("")

        plt.subplot(223)
        plt.title("Recurrence Interval")
        self.plot_RI_v_discharge()

        if self.save:
            plt.savefig(os.path.join(self.figuredir,
                                     "rivers", f"river_{self.site}.pdf"))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = ""
        string += f"Name:................{self.name}\n"
        string += f"Site #:..............{self.site}\n"
        string += f"Time-Range:..........{int(np.min(self.years))}-{int(np.max(self.years))}\n"
        string += f"Stage-Range:.........{int(np.min(self.stage))}-{int(np.max(self.stage))} [f]\n"
        string += f"Discharge-Range:.....{int(np.min(self.discharge))}-{int(np.max(self.discharge))} [f^3/s]\n"

        return string


if __name__ == "__main__":

    # River numbers

    stations = {
        "Lucas Sawade": '06817500',
        "Matthew Butler": '4127997',
        "William Carpenter": '1199000',
        "Kaila Carroll": '1209700',
        "Shannon Chaffers": '4174500',
        "Sophia Duchateau": '50075000',
        "Michael Folding": '1208990',
        "Lauren Huff": '2089500',
        "Caren Ju": '4102500',
        "Sarah Kamanzi": '1175670',
        "Elias Mosby": '8062000',
        "Jay Rolader": '2336300',
        "Isabel Segel": '3085500',
        "Yuxin Shi": '11023340',
        "Isra Thange": '4074950'
    }


# Get Stationname
with PdfPages(os.path.join(lpy.DOCFIGURES, f"rivers.pdf")) as pdf:
    for _student, _station in stations.items():
        r = River(_station, pre_title=_student, save=False)
        r.populate()
        r.plot_summary()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
