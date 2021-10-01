"https://nwis.waterdata.usgs.gov/nwis/dv?cb_00060=on&cb_00065=on&format=rdb&site_no=06817700&referred_module=sw&period=&begin_date=1987-04-01&end_date=2021-04-13"


import os
from copy import deepcopy
import lwsspy as lpy
import numpy as np
from matplotlib.dates import datestr2num
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
import matplotlib.dates as mdates


def naninfmin(a):
    return np.nanmin(a[np.abs(a) != np.inf])


def naninfmax(a):
    return np.nanmax(a[np.abs(a) != np.inf])


class River:
    """Class to download and plot data from USGS river data.

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.04.16 10.00

    """
    name: str
    site: str
    dates: np.ndarray
    discharge: np.ndarray
    stage: np.ndarray

    def __init__(self, stationnumber: str, flood_stage: float = None,
                 datadir: str = os.path.join(lpy.base.DOCFIGURESCRIPTDATA),
                 figuredir: str = os.path.join(lpy.base.DOCFIGURES),
                 pre_title: str = None, peak=True,
                 save: bool = False,
                 starttime: datetime = datetime(1900, 1, 1),
                 endtime: datetime = datetime.now(),
                 parameterdict: dict = dict(stage="00065", discharge="00060")):
        """Creates a river class using a stationnumber. The calls can populate
        itself by downloading and reading the tab separated file of USGS river 
        and ocean stations

        Parameters
        ----------
        stationnumber : str
            station or site no
        flood_stage : float, optional
            A number in feet the , by default None
        datadir : str, optional
            directory where to save the river files, by default os.path.join(lpy.DOCFIGURESCRIPTDATA)
        figuredir : str, optional
            directory where to save output figures, by default os.path.join(lpy.DOCFIGURES)
        pre_title : str, optional
            Plotted in bold without , by default None
        peak : bool, optional
            If True the file for Peak Annual discharge is downloaded, if False
            the parameterdict has to be populated. Currently supported 
            parameters are the stage="00065", and the discharge="00060",
            by default True
        save : bool, optional
            flag to set whether figures should be saved to a directory,
            by default False
        starttime : datetime, optional
            starttime for the data download, by default datetime(1900, 1, 1)
        endtime : datetime, optional
            endtime for the data download, by default datetime.now()
        parameterdict : dict, optional
            which parameters to download if peak is False, the default
            parameter are also the only ones supported right now.
            by default dict(stage="00065", discharge="00060")

        """

        # Download/station parameters
        self.site = stationnumber.zfill(8)
        self.flood_stage = flood_stage
        self.parameterdict = parameterdict  # Only relevant if peak is False
        self.peak = peak

        # From when to when to get data
        self.starttime = starttime
        self.endtime = endtime

        # Storage parameters
        self.datadir = datadir
        self.figuredir = figuredir

        # Plot parameters
        self.save = save
        self.pre_title = pre_title

        # Get url
        self.construct_url()

    def construct_url(self):
        """Constructs the URL to request the data for the given station"""

        # Base URL
        self.url = 'https://nwis.waterdata.usgs.gov/nwis/'

        # Decide whether to get parameters or just the peak annual discharge
        if self.peak:
            self.url += f"peak?"
        else:
            self.url += f"dv?"

            # Add each parameter
            for _, _parval in self.parameterdict.items():
                self.url += f"cb_{_parval}=on&"

        # Add site and agency id
        self.url += f'site_no={self.site}&agency_cd=USGS&format=rdb'

        # Add start and endtime
        datefmt = "%Y-%m-%d"
        self.url += f"&period=&begin_date={self.starttime.strftime(datefmt)}"
        self.url += f"&end_date={self.endtime.strftime(datefmt)}"

    def populate(self):
        """Main data population function. Downloads the data, saves it to
        the directory and loads the containing table and poulates the arrays.
        """

        # River file
        riverdir = os.path.join(self.datadir, "rivers")

        # Create data dir if it doesnt exist
        if os.path.exists(riverdir) is False:
            os.makedirs(riverdir)

        # File qualifier
        if self.peak:
            postfix = "peak"
        else:
            postfix = "dv"

        # Construct filename
        riverfile = os.path.join(
            riverdir, f"river_{self.site}_{postfix}.txt")

        # Download if I don't have it --> add overwrite flag file?
        if os.path.exists(riverfile) is False:
            lpy.downloadfile(self.url, riverfile)

        # Load file as string
        with open(riverfile, 'r') as f:
            riverstr = f.read()
            if 'No sites/data found using the selection criteria specified' in riverstr:
                raise ValueError(
                    "Double Check your site number, no data for the current one")

            # Get the name of the River
            self.name = self.get_station_name(riverstr)

            # Get the head of the station important, because the columns
            # are not in the same order as the URL
            self.header = self.get_station_header(riverstr)

        if self.peak:
            # If annual peak is the file we wanna get.
            _, dates, discharge, stage = self.get_station_table_peak(
                riverfile)

            # Remove dates inbetween start and endtime
            datepos = np.where((self.starttime <= dates)
                               & (dates <= self.endtime))
            dates = dates[datepos]
            discharge = discharge[datepos]
            stage = stage[datepos]

            # Create vector of years (want to add NaNs between years that
            # are )
            years = np.array([d.year for d in dates])
            pos = np.where(np.diff(years) > 1.0)[0]

            # Fix data gaps using the years with gaps
            if len(pos) > 0:
                self.dates = self.fix_date_gaps(dates, pos, date=True)
                self.discharge = self.fix_date_gaps(discharge, pos)
                self.stage = self.fix_date_gaps(stage, pos)
            else:
                self.dates = dates
                self.discharge = discharge.astype(float)
                self.stage = stage.astype(float)

        else:
            cols = self.get_data_columns()
            self.get_station_table_params(riverfile, cols)

        # Conversion of the dates to matplotlib dates for plotting
        self.mdates = np.array([mdates.date2num(_i) for _i in self.dates])

    @staticmethod
    def fix_date_gaps(x, pos, date=False):
        # List vs. numpy array management
        if type(x) is list:
            y = deepcopy(x)
        else:
            y = x.tolist()

        # Actual insertion
        for _p in pos[::-1]:
            y.insert(_p+1, np.nan)

        return np.array(y)

    def get_station_name(self, riverstr: str):
        lines = riverstr.split('\n')
        for _line in lines:
            # Break if going past comments...
            if _line[0] != '#':
                raise ValueError("Station name not found!")
                break

            # Get name of stations from the comment
            if f"USGS {self.site}" in _line:
                return ' '.join(_line.split()[1:])

    def get_station_header(self, riverstr):
        lines = riverstr.split('\n')
        for _line in lines:
            # Break if going past comments...
            if _line[0] == '#':
                pass
            elif "agency_cd" in _line:
                return _line.split()

    def get_data_columns(self):
        cols = dict()
        for _i, string in enumerate(self.header):
            spstr = string.split("_")
            if len(spstr) == 3:
                par = [p for p, v in self.parameterdict.items()
                       if v == spstr[1]][0]
                cols[par] = _i
        return cols

    @ staticmethod
    def str2date(x: str):
        x = x.split('-')
        if x[1] == "00":
            x[1] = "01"
        if x[2] == "00":
            x[2] = "01"
        return datetime.fromisoformat(f"{x[0]}-{x[1]}-{x[2]}")

    def get_station_table_params(self, riverfile, cols):
        """Takes in the location of the tab-separated river data file and
        outputs and populates the corresponding attributes of the river class.
        """

        # Read file
        # Columns are
        # agency_cd	site_no	peak_dt	peak_tm	peak_va	peak_cd	gage_ht	gage_ht_cd
        # Comments removes the header underneath the comments
        comments = ["#", 'agency', '5s']
        strarray = np.loadtxt(
            riverfile, dtype=str, comments=comments, delimiter='\t')

        self.dates = np.array([self.str2date(x) for x in strarray[:, 2]])
        for _par in self.parameterdict.keys():
            setattr(self, _par, np.array(
                [float(x) if x != '' else np.nan for x in strarray[:, cols[_par]]]))

    def get_station_table_peak(self, riverfile):
        """Takes in the location of the tab-separated river data file and outputs
        tuple of arrays containing the.
        (siteno, year, discharge, stage)
        """

        # Read file
        # Columns are
        # agency_cd	site_no	peak_dt	peak_tm	peak_va	peak_cd	gage_ht	gage_ht_cd
        # Comments removes the header underneath the comments
        comments = ["#", 'agency', '5s']
        strarray = np.loadtxt(
            riverfile, dtype=str, comments=comments, delimiter='\t')

        site = strarray[0, 1]
        dates = np.array([self.str2date(x) for x in strarray[:, 2]])
        discharge = np.array(
            [float(x) if x != '' else np.nan for x in strarray[:, 4]])
        stage = np.array(
            [float(x) if x != '' else np.nan for x in strarray[:, 6]])

        return site, dates, discharge, stage

    def plot_peak_annual_discharge(self):
        ax = plt.gca()
        plt.plot(self.mdates, self.discharge, ".-k")
        plt.xlabel(r'Year')
        plt.ylabel(r'Discharge [f$^3$/s]')
        plt.grid('on')

        # Add flood stage line
        if self.flood_stage is not None:
            # Compute relationship
            istage = np.argsort(self.stage)
            T = lpy.Trendline(self.stage[istage], self.discharge[istage])
            T.fit()

            # Plot floodstage line
            plt.plot(
                (naninfmin(self.mdates), naninfmax(self.mdates)),
                (T.model(self.flood_stage), T.model(self.flood_stage)),
                label="Flood Stage")
            plt.legend()

        # Set limits
        ax.set_xlim((naninfmin(self.mdates), naninfmax(self.mdates)))
        ax.set_ylim((0, 1.25*naninfmax(self.discharge)))

        # Set x axis date formatting
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
        ax.xaxis.set_major_locator(loc)

    def plot_stage_v_discharge(self):

        # Stage Trendline
        istage = np.argsort(self.stage)
        T = lpy.Trendline(
            self.stage[istage], self.discharge[istage], pred_type='2')
        T.fit()
        x_trend = np.linspace(naninfmin(self.stage),
                              naninfmax(self.stage), 1000)

        ax = plt.gca()
        plt.plot(self.stage, self.discharge, ".k")
        T.plot_trendline(x_trend, 'k', alpha=0.75)
        T.plot_confidence_band(
            x_trend, ec='none', fc='lightgrey', zorder=0)
        plt.xlabel(r'Stage [f]')
        plt.ylabel(r'Discharge [f$^3$/s]')
        plt.grid('on')
        lpy.plot_label(ax, fr'$R^2 = {T.R2:4.2f}$',
                       location=1, fontdict=dict(fontsize='small'))

        # Set limits
        ax.set_xlim((naninfmin(self.stage),  naninfmax(self.stage)))
        ax.set_ylim((0, 1.25*naninfmax(self.discharge)))

    def plot_stage_v_discharge_evo(self, **kwargs):

        # Get axes
        ax = plt.gca()

        # Get nonnan positions
        pos = ~np.isnan(self.discharge) & ~np.isnan(self.stage)

        # Plot in scatter plot
        sc = ax.scatter(self.stage[pos], self.discharge[pos],
                        c=self.mdates[pos],
                        s=5, cmap='rainbow', **kwargs)

        # Set axis limits
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)

        plt.xlabel("Gage Height [f]")
        plt.ylabel("Discharge [f$^3$]")
        plt.title(
            fr"Evolution of the Stage-Discharge Relation")
        loc = mdates.AutoDateLocator()
        cbar = lpy.nice_colorbar(sc, aspect=40, ticks=loc, fraction=0.05,
                                 format=mdates.AutoDateFormatter(loc))

    def plot_RI_v_discharge(self):

        # Get only values that don't have a nan
        pos = ~np.isnan(self.mdates) & ~np.isnan(self.discharge)
        discharge = self.discharge[pos]
        dates = self.dates[pos]
        md = self.mdates[pos]
        N = len(md)
        preRI = np.arange(N+1, 1, -1)

        # Sort values after discharge
        idischarge = np.argsort(discharge)

        # Sort values after year
        iyear = np.argsort(dates)
        iyeardischarge = np.argsort(iyear[idischarge])

        # Compute RI
        self.RI = ((N+1)/preRI[iyeardischarge])
        self.nonnandischarge = discharge
        iRI = np.argsort(self.RI)

        # Compute Trendline
        T = lpy.Trendline(self.RI[iRI], discharge[iRI], pred_type='log')
        T.fit()
        discharge_predict_100 = T.model(100.0)

        # Plot RI and its 100yr prediction
        ax = plt.gca()
        plt.plot(self.RI, discharge, ".k")
        T.plot_trendline(np.linspace(1, 100, 1000), "k", alpha=0.75)
        T.plot_confidence_band(np.linspace(1, 100, 1000),
                               ec='none', fc='lightgrey', zorder=0)

        # Plot identifier
        plt.plot((100, 100), (0, 1.25 * discharge_predict_100),
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
        ax.set_ylim(bottom=0, top=1.25 * discharge_predict_100)

    def plot_summary(self):

        fig = plt.figure(figsize=(12, 10))
        plt.subplots_adjust(wspace=0.225, hspace=0.3)

        if self.pre_title is None:
            plt.suptitle(self.name)
        else:
            plt.suptitle(fr"$\bf{{{self.pre_title}}}$: " + self.name)

        plt.subplot(221)
        plt.title("Peak Annual Discharge")
        self.plot_peak_annual_discharge()

        ax = plt.subplot(222)
        plt.title("Stage v Discharge Relationship")
        self.plot_stage_v_discharge()
        plt.ylabel("")

        plt.subplot(223)
        plt.title("Recurrence Interval")
        self.plot_RI_v_discharge()

        plt.subplot(224)
        self.plot_stage_v_discharge_evo()

        if self.save:
            plt.savefig(os.path.join(self.figuredir,
                                     "rivers", f"river_{self.site}.pdf"))

    def convert_to_annual_peak_discharge(self):

        minyear = np.min(self.dates).year
        maxyear = np.max(self.dates).year

        # Create arrays
        years = np.array([datetime(_i, 1, 1)
                          for _i in range(minyear, maxyear+2)])
        discharge = np.zeros(len(years)-1, dtype=float)
        stage = np.zeros(len(years)-1, dtype=float)

        for _i in range(len(years)-1):

            pos = np.where((years[_i] < self.dates) &
                           (self.dates < years[_i+1]))[0]

            if len(pos) > 0:
                iyearmax = np.argmax(self.discharge[pos])
                discharge[_i] = self.discharge[pos][iyearmax]
                stage[_i] = self.stage[pos][iyearmax]

            else:
                discharge[_i] = np.nan
                stage[_i] = np.nan

        # Create vector of years
        intyears = np.array([d.year for d in years])
        pos = np.where(np.diff(intyears) > 1.0)[0]

        # Update the stuff
        self.discharge = self.fix_date_gaps(discharge, pos)
        self.stage = self.fix_date_gaps(stage, pos)
        self.dates = self.fix_date_gaps(years[:-1], pos)

        for date in years:
            print(date)
        # Conversion of the dates to matplotlib dates for plotting
        self.mdates = np.array([mdates.date2num(_i) for _i in self.dates])

    def pop(self, i):
        """Pop single index from the arrays. Useful to get rid of outliers.
        Santa Ana example."""

        self.dates = np.delete(self.dates, i)
        self.mdates = np.delete(self.mdates, i)
        self.discharge = np.delete(self.discharge, i)
        self.stage = np.delete(self.stage, i)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = ""
        string += f"Name:................{self.name}\n"
        string += f"Site #:..............{self.site}\n"
        string += f"Time-Range:..........{int(np.min(self.dates))}-{int(np.max(self.dates))}\n"
        string += f"Stage-Range:.........{int(np.min(self.stage))}-{int(np.max(self.stage))} [f]\n"
        string += f"Discharge-Range:.....{int(np.min(self.discharge))}-{int(np.max(self.discharge))} [f^3/s]\n"

        return string
