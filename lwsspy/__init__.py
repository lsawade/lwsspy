import sys
import os

DOCFIGURES = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))),
    'docs', 'source', 'chapters', 'figures')

# Global imports when running the modules for testing
if "-m" not in sys.argv:

    # -------- CONSTANTS ------------------------------------------------------
    from .constants import EARTH_RADIUS  # noqa

    # -------- FUNCTIONS & CLASSES --------------------------------------------

    # Inversion
    from .inversion.optimizer import Optimization  # noqa
    from .inversion.plot_optimization import plot_optimization  # noqa
    from .inversion.plot_model_history import plot_model_history  # noqa

    # IO
    from .utils.io import load_asdf  # noqa
    from .utils.io import load_json  # noqa
    from .utils.io import dump_json  # noqa
    from .utils.io import read_yaml_file  # noqa
    from .utils.io import write_yaml_file  # noqa
    from .utils.output import nostdout  # noqa
    from .utils.output import print_action  # noqa
    from .utils.output import print_bar  # noqa
    from .utils.output import print_section  # noqa

    # Math
    from .math.convm import convm  # noqa
    from .math.eigsort import eigsort  # noqa
    from .math.magnitude import magnitude  # noqa
    from .math.SphericalNN import SphericalNN  # noqa

    # Plot
    from .plot_util.updaterc import updaterc  # noqa
    from .plot_util.updaterc import updaterc_pres  # noqa
    from .plot_util.remove_ticklabels import remove_xticklabels  # noqa
    from .plot_util.remove_ticklabels import remove_yticklabels  # noqa
    from .plot_util.figcolorbar import figcolorbar  # noqa

    # Seismology
    from .seismo.cmt2inv import cmt2inv  # noqa
    from .seismo.cmt2stationxml import cmt2stationxml  # noqa
    from .seismo.cmtdir2stationxmldir import cmtdir2stationxmldir  # noqa
    from .seismo.download_waveforms_cmt2storage import download_waveforms_cmt2storage  # noqa
    from .seismo.download_waveforms_to_storage import download_waveforms_to_storage  # noqa
    from .seismo.filterstationxml import filterstationxml  # noqa
    from .seismo.inv2stationxml import inv2stationxml  # noqa
    from .seismo.perturb_cmt import perturb_cmt  # noqa
    from .seismo.perturb_cmt import perturb_cmt_dir  # noqa
    from .seismo.plot_stationxml import plot_station_xml  # noqa
    from .seismo.process.process import process_stream  # noqa
    from .seismo.process.process_wrapper import process_wrapper  # noqa
    from .seismo.process.rotate import rotate_stream  # noqa
    from .seismo.read_inventory import flex_read_inventory as read_inventory  # noqa
    from .seismo.source import CMTSource  # noqa
    from .seismo.validate_cmt import validate_cmt  # noqa
    from .seismo.specfem.cmt2rundir import cmt2rundir  # noqa
    from .seismo.specfem.cmt2simdir import cmt2simdir  # noqa
    from .seismo.specfem.cmt2STATIONS import cmt2STATIONS  # noqa
    from .seismo.specfem.cmtdir2rundirs import cmtdir2rundirs  # noqa
    from .seismo.specfem.cmtdir2simdirs import cmtdir2simdirs  # noqa
    from .seismo.specfem.createsimdir import createsimdir  # noqa
    from .seismo.specfem.getsimdirSTATIONS import getsimdirSTATIONS  # noqa
    from .seismo.specfem.inv2STATIONS import inv2STATIONS  # noqa
    from .seismo.specfem.read_parfile import read_parfile  # noqa
    from .seismo.specfem.stationxml2STATIONS import stationxml2STATIONS  # noqa
    from .seismo.specfem.stationxmldir2STATIONSdir import stationxmldir2STATIONSdir  # noqa
    from .seismo.specfem.write_parfile import write_parfile  # noqa
    from .seismo.window.window import window_on_stream  # noqa
    from .seismo.window.add_tapers import add_tapers  # noqa
    from .seismo.window.stream_cost_win import stream_cost_win  # noqa
    from .seismo.window.stream_grad_frechet_win import stream_grad_frechet_win  # noqa
    from .seismo.window.stream_grad_hess_frechet_win import stream_grad_and_hess_win  # noqa

    # Shell
    from .shell.cat import cat  # noqa
    from .shell.copy_dirtree import copy_dirtree  # noqa
    from .shell.cp import cp  # noqa
    from .shell.create_dirtree import create_dirtree  # noqa
    from .shell.cpdir import cpdir  # noqa
    from .shell.ln import ln  # noqa
    from .shell.readfile import readfile  # noqa
    from .shell.run_cmds_parallel import run_cmds_parallel  # noqa
    from .shell.touch import touch  # noqa
    from .shell.writefile import writefile  # noqa

    # Statistics
    from .statistics.clm import clm  # noqa
    from .statistics.distlist import distlist  # noqa
    from .statistics.errorellipse import errorellipse  # noqa
    from .statistics.fakerelation import fakerelation  # noqa
    from .statistics.fitgaussian2d import fitgaussian2d  # noqa
    from .statistics.gaussian2d import gaussian2d  # noqa
    from .statistics.normalstdheight import normalstdheight  # noqa

    # Weather
    from .weather.requestweather import requestweather  # noqa
    from .weather.weather import weather  # noqa
    from .weather.drop2pickle import drop2pickle  # noqa
