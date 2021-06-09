import sys
import os
import platform

DOCFIGURES: str = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))),
    'docs', 'source', 'chapters', 'figures')
DOCFIGURESCRIPTDATA: str = os.path.join(DOCFIGURES, 'scripts', 'data')

DOWNLOAD_CACHE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'download_cache')

CONSTANT_DATA: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'constant_data')

GCMT_DATA: str = os.path.join(CONSTANT_DATA, 'gcmt')

# Global imports when running the modules for testing
if "-m" not in sys.argv:

    # -------- CONSTANTS ------------------------------------------------------
    from .constants import EARTH_RADIUS_M  # noqa
    from .constants import EARTH_RADIUS_KM  # noqa
    from .constants import EARTH_CIRCUM_M  # noqa
    from .constants import EARTH_CIRCUM_KM  # noqa
    from .constants import EMC_DATABASE  # noqa
    from .constants import DEG2KM  # noqa
    from .constants import KM2DEG  # noqa
    from .constants import DEG2M  # noqa
    from .constants import M2DEG  # noqa
    from .constants import abc  # noqa

    # -------- FUNCTIONS & CLASSES --------------------------------------------

    # IO
    from .utils.io import dump_json  # noqa
    # from .utils.io import load_asdf  # noqa
    from .utils.io import load_json  # noqa
    from .utils.io import loadxy_csv  # noqa
    from .utils.io import loadmat  # noqa
    from .utils.io import read_yaml_file  # noqa
    from .utils.io import smart_read_yaml  # noqa
    from .utils.io import is_mpi_env  # noqa
    from .utils.io import write_yaml_file  # noqa

    # STDOUT
    from .utils.customlogformatter import CustomFormatter  # noqa
    from .utils.output import nostdout  # noqa
    from .utils.output import print_action  # noqa
    from .utils.output import print_bar  # noqa
    from .utils.output import print_section  # noqa
    from .utils.output import log_action  # noqa
    from .utils.output import log_bar  # noqa
    from .utils.output import log_section  # noqa

    # Geographical things
    from .geo.geo_weights import GeoWeights  # noqa
    from .geo.azi_weights import azi_weights  # noqa

    # Inversion
    from .inversion.optimizer import Optimization  # noqa
    from .inversion.plot_optimization import plot_optimization  # noqa
    from .inversion.plot_model_history import plot_model_history  # noqa
    from .inversion.plot_single_parameter_optimization import plot_single_parameter_optimization  # noqa

    # Maps
    from .maps.fix_map_extent import fix_map_extent  # noqa
    from .maps.line_buffer import line_buffer  # noqa
    from .maps.map_axes import map_axes  # noqa
    from .maps.plot_map import plot_map  # noqa
    # from .maps.plot_topography import plot_topography  # noqa
    from .maps.plot_line_buffer import plot_line_buffer  # noqa
    # from .maps.plot_litho import plot_litho  # noqa
    # from .maps.read_etopo import read_etopo  # noqa
    # from .maps.read_litho import read_litho  # noqa
    from .maps.reckon import reckon  # noqa
    from .maps.topocolormap import topocolormap  # noqa
    from .maps.topography_design import TopographyDesign  # noqa
    from .maps.gctrack import gctrack  # noqa

    print('past maps')
    # Math
    from .math.cart2sph import cart2sph  # noqa
    from .math.convm import convm  # noqa
    from .math.eigsort import eigsort  # noqa
    from .math.envelope import envelope  # noqa
    from .math.geo2cart import geo2cart  # noqa
    from .math.cart2geo import cart2geo  # noqa
    from .math.cart2sph import cart2sph  # noqa
    from .math.cart2pol import cart2pol  # noqa
    from .math.magnitude import magnitude  # noqa
    from .math.pol2cart import pol2cart  # noqa
    from .math.project2D import project2D  # noqa
    from .math.rodrigues import rodrigues  # noqa
    from .math.Ra2b import Ra2b  # noqa
    from .math.sph2cart import sph2cart  # noqa
    from .math.SphericalNN import SphericalNN  # noqa
    print('past math')
    # Plot
    from .plot_util.figcolorbar import figcolorbar  # noqa
    from .plot_util.fixedpointcolornorm import FixedPointColorNorm  # noqa
    from .plot_util.get_aspect import get_aspect  # noqa
    from .plot_util.get_stats_label_length import get_stats_label_length  # noqa
    from .plot_util.get_continuous_cmap import get_continuous_cmap  # noqa
    from .plot_util.hex2rgb import hex_to_rgb  # noqa
    from .plot_util.hudsondiamondaxes import hdaxes  # noqa
    from .plot_util.midpointcolornorm import MidpointNormalize  # noqa
    from .plot_util.midpointlognorm import MidPointLogNorm  # noqa
    from .plot_util.multiple_formatter import multiple_formatter  # noqa
    from .plot_util.multiple_formatter import Multiple  # noqa
    from .plot_util.nice_colorbar import nice_colorbar  # noqa
    from .plot_util.pick_colors_from_cmap import pick_colors_from_cmap  # noqa
    from .plot_util.pick_data_from_image import pick_data_from_image  # noqa
    from .plot_util.plot_label import plot_label  # noqa
    from .plot_util.plot_xyz_line import plot_xyz_line  # noqa
    from .plot_util.pz_figure import pz_figure  # noqa
    from .plot_util.remove_ticklabels import remove_xticklabels  # noqa
    from .plot_util.remove_ticklabels import remove_yticklabels  # noqa
    from .plot_util.remove_ticklabels import remove_ticklabels  # noqa
    from .plot_util.remove_ticklabels import remove_ticks  # noqa
    from .plot_util.remove_ticklabels import remove_ticklabels_bottomleft  # noqa
    from .plot_util.remove_ticklabels import remove_ticklabels_topright  # noqa
    from .plot_util.rgb2dec import rgb_to_dec  # noqa
    from .plot_util.right_align_legend import right_align_legend  # noqa
    from .plot_util.scatter_hist import scatter_hist  # noqa
    from .plot_util.smooth_nan_image import smooth_nan_image  # noqa
    from .plot_util.trendline import Trendline  # noqa
    from .plot_util.updaterc import updaterc  # noqa
    from .plot_util.updaterc import updaterc_pres  # noqa
    from .plot_util.update_colorcycler import update_colorcycler  # noqa
    from .plot_util.view_colormap import view_colormap  # noqa
    from .plot_util.zerotraceaxes import ztaxes  # noqa

    print('past plotutil')
    # Seismology
    from .seismo.cmt2inv import cmt2inv  # noqa
    from .seismo.cmt2stationxml import cmt2stationxml  # noqa
    from .seismo.cmtdir2stationxmldir import cmtdir2stationxmldir  # noqa
    from .seismo.cmt_catalog import CMTCatalog  # noqa
    from .seismo.compare_catalogs import CompareCatalogs  # noqa
    from .seismo.costgradhess import CostGradHess  # noqa
    from .seismo.costgradhess_log import CostGradHessLogEnergy  # noqa
    from .seismo.download_data import download_data  # noqa
    from .seismo.download_gcmt_catalog import download_gcmt_catalog  # noqa
    from .seismo.download_waveforms_cmt2storage import download_waveforms_cmt2storage  # noqa
    from .seismo.download_waveforms_to_storage import download_waveforms_to_storage  # noqa
    from .seismo.filterstationxml import filterstationxml  # noqa
    from .seismo.gaussiant import gaussiant  # noqa
    from .seismo.gaussiant import dgaussiant  # noqa
    from .seismo.get_inv_aspect_extent import get_inv_aspect_extent  # noqa
    from .seismo.instaseis.get_prem20s import get_prem20s  # noqa
    from .seismo.inv2geoloc import inv2geoloc  # noqa
    from .seismo.inv2net_sta import inv2net_sta  # noqa
    from .seismo.inv2stationxml import inv2stationxml  # noqa
    from .seismo.m0_2_mw import m0_2_mw  # noqa
    from .seismo.perturb_cmt import perturb_cmt  # noqa
    from .seismo.perturb_cmt import perturb_cmt_dir  # noqa
    from .seismo.plot_stationxml import plot_station_xml  # noqa
    from .seismo.plot_traveltimes import compute_traveltimes  # noqa
    from .seismo.plot_traveltimes import plot_traveltimes  # noqa
    from .seismo.plot_traveltimes_ak135 import plot_traveltimes_ak135  # noqa
    from .seismo.plot_inventory import plot_inventory  # noqa
    from .seismo.plot_quakes import plot_quakes  # noqa
    from .seismo.plot_seismogram import plot_seismogram  # noqa
    from .seismo.plot_seismogram import plot_seismogram_by_station  # noqa
    from .seismo.plot_seismogram import plot_seismograms  # noqa
    from .seismo.stream_pdf import stream_pdf  # noqa
    from .seismo.process.process import process_stream  # noqa
    print('past some point')
    from .seismo.process.mpiprocessclass import MPIProcessStream  # noqa
    from .seismo.process.multiprocess_stream import multiprocess_stream  # noqa
    from .seismo.process.process_wrapper import process_wrapper  # noqa
    from .seismo.process.rotate import rotate_stream  # noqa
    from .seismo.process.process_classifier import ProcessParams  # noqa
    from .seismo.process.process_classifier import filter_scaling  # noqa
    from .seismo.read_gcmt_catalog import read_gcmt_catalog  # noqa
    from .seismo.read_inventory import flex_read_inventory as read_inventory  # noqa
    from .seismo.source import CMTSource  # noqa
    from .seismo.stream_multiply import stream_multiply  # noqa
    from .seismo.validate_cmt import validate_cmt  # noqa
    from .seismo.specfem.cmt2rundir import cmt2rundir  # noqa
    from .seismo.specfem.cmt2simdir import cmt2simdir  # noqa
    from .seismo.specfem.cmt2STATIONS import cmt2STATIONS  # noqa
    from .seismo.specfem.cmtdir2rundirs import cmtdir2rundirs  # noqa
    from .seismo.specfem.cmtdir2simdirs import cmtdir2simdirs  # noqa
    from .seismo.specfem.createsimdir import createsimdir  # noqa
    from .seismo.specfem.getsimdirSTATIONS import getsimdirSTATIONS  # noqa
    from .seismo.specfem.inv2STATIONS import inv2STATIONS  # noqa
    from .seismo.specfem.plot_csv_depth_slice import plot_csv_depth_slice  # noqa
    from .seismo.specfem.plot_specfem_xsec_depth import plot_specfem_xsec_depth  # noqa
    from .seismo.specfem.read_parfile import read_parfile  # noqa
    from .seismo.specfem.read_specfem_xsec_depth import read_specfem_xsec_depth  # noqa
    from .seismo.specfem.stationxml2STATIONS import stationxml2STATIONS  # noqa
    from .seismo.specfem.stationxmldir2STATIONSdir import stationxmldir2STATIONSdir  # noqa
    from .seismo.specfem.write_parfile import write_parfile  # noqa
    from .seismo.window.multiwindow_stream import multiwindow_stream  # noqa
    from .seismo.window.mpiwindowclass import MPIWindowStream  # noqa
    from .seismo.window.window import window_on_stream  # noqa
    from .seismo.window.window import merge_trace_windows  # noqa
    from .seismo.window.add_tapers import add_tapers  # noqa
    from .seismo.window.stream_cost_win import stream_cost_win  # noqa
    from .seismo.window.stream_grad_frechet_win import stream_grad_frechet_win  # noqa
    from .seismo.window.stream_grad_hess_win import stream_grad_and_hess_win  # noqa
    from .seismo.read_gcmt_data import load_1976_2004_mag  # noqa
    from .seismo.read_gcmt_data import load_2004_2010_mag  # noqa
    from .seismo.read_gcmt_data import load_num_events  # noqa
    from .seismo.read_gcmt_data import load_cum_mag  # noqa

    print('past seismo')

    # CMT3D
    from .seismo.invertcmt.GCMT3DInversion import GCMT3DInversion  # noqa
    from .seismo.invertcmt.plot_weights import plot_weightpickle  # noqa
    from .seismo.invertcmt.plot_weights import plot_weights  # noqa
    from .seismo.invertcmt.plot_measurements import get_database_measurements  # noqa
    from .seismo.invertcmt.plot_measurements import plot_measurements  # noqa
    from .seismo.invertcmt.plot_measurements import plot_measurement_pkl  # noqa
    from .seismo.invertcmt.plot_measurements import compute_window_hists  # noqa
    from .seismo.invertcmt.plot_measurements import plot_window_histograms  # noqa
    from .seismo.invertcmt.process_classifier import filter_scaling  # noqa
    from .seismo.invertcmt.process_classifier import ProcessParams  # noqa
    from .seismo.invertcmt.io import read_output_traces  # noqa
    from .seismo.invertcmt.io import read_measurements  # noqa
    # Shell
    from .shell.cat import cat  # noqa
    from .shell.copy_dirtree import copy_dirtree  # noqa
    from .shell.cp import cp  # noqa
    from .shell.create_dirtree import create_dirtree  # noqa
    from .shell.cpdir import cpdir  # noqa
    from .shell.downloadfile import downloadfile  # noqa
    from .shell.downloadfile import download_threaded  # noqa
    from .shell.get_url_paths import get_url_paths  # noqa
    from .shell.ln import ln  # noqa
    from .shell.readfile import readfile  # noqa
    from .shell.run_cmds_parallel import run_cmds_parallel  # noqa
    from .shell.touch import touch  # noqa
    from .shell.unzip import unzip  # noqa
    from .shell.ungzip import ungzip  # noqa
    from .shell.writefile import writefile  # noqa

    # Simple Signal processing functions
    from .signal.dlna import dlna  # noqa
    from .signal.xcorr import xcorr, correct_window_index  # noqa
    from .signal.norm import norm1, norm2, dnorm1, dnorm2  # noqa
    from .signal.power import power_l1, power_l2  # noqa

    # Statistics
    from .statistics.clm import clm  # noqa
    from .statistics.distlist import distlist  # noqa
    from .statistics.errorellipse import errorellipse  # noqa
    from .statistics.even2Dpoints import even2Dpoints  # noqa
    from .statistics.fakerelation import fakerelation  # noqa
    from .statistics.fitgaussian2d import fitgaussian2d  # noqa
    from .statistics.gaussian2d import gaussian2d  # noqa
    from .statistics.normalstdheight import normalstdheight  # noqa

    # Utilities
    from .utils.add_years import add_years  # noqa
    from .utils.cpu_count import cpu_count  # noqa
    from .utils.chunks import chunks  # noqa
    from .utils.date2year import date2year  # noqa
    from .utils.get_unique_lists import get_unique_lists  # noqa
    from .utils.increase_fontsize import increase_fontsize  # noqa
    from .utils.multiwrapper import poolcontext  # noqa
    from .utils.multiwrapper import starmap_with_kwargs  # noqa
    from .utils.pixels2data import pixels2data  # noqa
    from .utils.reduce_fontsize import reduce_fontsize  # noqa
    from .utils.sec2hhmmss import sec2hhmmss  # noqa
    from .utils.sec2hhmmss import sec2timestamp  # noqa
    from .utils.threadwork import threadwork  # noqa
    from .utils.timer import Timer  # noqa
    from .utils.year2date import year2date  # noqa

    # Weather
    from .weather.requestweather import requestweather  # noqa
    from .weather.weather import weather  # noqa
    from .weather.river import River  # noqa
    from .weather.drop2pickle import drop2pickle  # noqa
