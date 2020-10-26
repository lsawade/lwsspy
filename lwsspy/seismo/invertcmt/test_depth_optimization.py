"""
This is the first script to enable depth iterations for the
CMTSOLUTTION depth.
"""
# %% Create inversion directory

# External
import os
import sys
import contextlib
import logging
from copy import deepcopy
from typing import Union
from subprocess import Popen, PIPE
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pyflex
from obspy.core.event.event import Event
from obspy.core.inventory.station import Station
from obspy import read, read_events, Stream, Trace, Inventory
from obspy.imaging.beachball import beach

# Internal
from lwsspy import CMTSource
from lwsspy import createsimdir
from lwsspy import process_stream
from lwsspy import stationxml2STATIONS
from lwsspy import read_parfile
from lwsspy import write_parfile
from lwsspy import read_yaml_file
from lwsspy import read_inventory
from lwsspy import updaterc
from lwsspy import Optimization
from lwsspy.inversion.plot_optimization import plot_optimization
from lwsspy.inversion.plot_model_history import plot_model_history
from lwsspy import plot_station_xml
from lwsspy import nostdout
updaterc()
pyflex.logger.setLevel(logging.WARNING)

# %% window functions


def window_on_trace(obs: Trace, syn: Trace, config: pyflex.Config,
                    station: Union[pyflex.Station, Station, None] = None,
                    event: Union[pyflex.Event, Event, None] = None,
                    _verbose=False, figure_mode=False, figure_dir=None,
                    figure_format="pdf"):
    """
    Window selection on a trace(obspy.Trace)

    :param observed: observed trace
    :type observed: obspy.Trace
    :param synthetic: synthetic trace
    :type synthetic: obspy.Trace
    :param config: window selection config
    :type config_dict: pyflex.Config
    :param station: station information which provids station location to
        calculate the epicenter distance
    :type station: obspy.Inventory or pyflex.Station
    :param event: event information, providing the event information
    :type event: pyflex.Event, obspy.Catalog or obspy.Event
    :param figure_mode: output figure flag
    :type figure_mode: bool
    :param figure_dir: output figure directory
    :type figure_dir: str
    :param _verbose: verbose flag
    :type _verbose: bool
    :return:
    """
    if not isinstance(obs, Trace):
        raise ValueError("Input obs_tr should be obspy.Trace")
    if not isinstance(syn, Trace):
        raise ValueError("Input syn_tr should be obspy.Trace")
    if not isinstance(config, pyflex.Config):
        raise ValueError("Input config should be pyflex.Config")

    ws = pyflex.WindowSelector(obs, syn, config,
                               event=event, station=station)
    try:
        windows = ws.select_windows()
    except Exception as err:
        print(f"Error({obs.id}): {err}")
        windows = []

    # if figure_mode:
    #     plot_window_figure(figure_dir, obs.id, ws, _verbose,
    #                        figure_format=figure_format)
    if _verbose:
        print("Station %s picked %i windows" % (obs.id, len(windows)))

    return windows


def window_on_stream(observed: Stream, synthetic: Stream,
                     config_dict: dict,
                     station: Union[None, Inventory, pyflex.Station] = None,
                     event: Union[None, Event, pyflex.Event] = None,
                     figure_mode=False, figure_dir=None, _verbose=False):
    """
    Window selection on a Stream

    :param observed: observed stream
    :type observed: obspy.Stream
    :param synthetic: synthetic stream
    :type synthetic: obspy.Stream
    :param config_dict: window selection config dictionary, for example,
        {"Z": pyflex.Config, "R": pyflex.Config, "T": pyflex.Config}
    :type config_dict: dict
    :param station: station information which provids station location to
        calculate the epicenter distance
    :type station: obspy.Inventory or pyflex.Station
    :param event: event information, providing the event information
    :type event: pyflex.Event, obspy.Catalog or obspy.Event
    :param user_modules: user_module strings in a dict similar to config_dict.
    :type user_modules: dict
    :param figure_mode: output figure flag
    :type figure_mode: bool
    :param figure_dir: output figure directory
    :type figure_dir: str
    :param _verbose: verbose flag
    :type _verbose: bool
    :return:
    """
    if not isinstance(observed, Stream):
        raise ValueError("Input observed should be obspy.Stream")
    if not isinstance(synthetic, Stream):
        raise ValueError("Input synthetic should be obspy.Stream")
    if not isinstance(config_dict, dict):
        raise ValueError("Input config_dict should be dict")

    config_base = config_dict["config"]

    # all_windows = {}

    for component in config_dict["components"].keys():

        # Get component specific values
        config = deepcopy(config_base)
        if config_dict["components"][component] is not None:
            config.update(config_dict["components"][component])
        pf_config = pyflex.Config(**config)

        # Get single compenent of stream to work on it
        obs = observed.select(component=component)

        # Loop over traces
        for obs_tr in obs:
            # component = obs_tr.stats.channel[-1]
            try:
                syn_tr = synthetic.select(station=obs_tr.stats.station,
                                          network=obs_tr.stats.network,
                                          component=component)[0]
            except Exception as err:
                print("Couldn't find corresponding synt for obsd trace(%s):"
                      "%s" % (obs_tr.id, err))
                continue

            # Station is the normal inventory, nothing fancy
            # event is an ObsPy Event
            obs_tr.stats.windows = window_on_trace(
                obs_tr, syn_tr, pf_config, station=station,
                event=event, _verbose=_verbose,
                figure_mode=figure_mode, figure_dir=figure_dir)


def taper_windows(observed: Stream, taper_type: str = "tukey",
                  alpha: float = 0.25):

    if taper_type == "tukey":
        taper = signal.tukey

    for tr in observed:
        # Create empty list of tapers
        tr.stats.tapers = []

        for win in tr.stats.windows:
            length = win.right - win.left
            tr.stats.tapers.append(taper(length, alpha=alpha))


def merge_windows(observed: Stream, synthetic: Union[Stream, None] = None):
    """
    Keep only location ("00", "01", etc.) with the highest number of windows.
    """
    # new_windows = {}
    keepstream = Stream()
    for _i, tr in observed:
        try:
            network = tr.stats.network
            station = tr.stats.station
            channel = tr.stats.channel

            tmp_st = observed.select(network=network, station=station,
                                     channel=channel)

            # This checks which location of the Traces has the maximum amount
            # of windows
            if len(tmp_st) != 1:
                maxwintrace = tmp_st[0]
                for _j, tr2 in enumerate(tmp_st):
                    if _j != 0:
                        if len(tr2.windows) > len(tmp_st[_j-1]):
                            maxwintrace = tr2
                if tr == maxwintrace:
                    if len(tr.windows) == 0:
                        pass
                    else:
                        keepstream.append(tr)
        except Exception as e:
            print(f"Error at Trace {tr.get_id()}: {e}")

    return keepstream


# %% Function to launch specfem instances in parallel


def run_cmds_parallel(cmd_list, cwdlist=None):
    """Takes in a listt of shell commands:

    Parameters
    ----------
    cmd_list : list
        List of list of arguments

    Last modified: Lucas Sawade, 2020.09.18 19.00 (lsawade@princeton.edu)
    """

    # Create list of processes that immediately start execution
    if cwdlist is None:
        cwdlist = len(cmd_list) * None
    process_list = [Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd)
                    for cmd, cwd in zip(cmd_list, cwdlist)]

    # Wait for the processes to finish
    for proc in process_list:
        proc.wait()

    # Print RETURNCODE, STDOUT and STDERR
    for proc in process_list:
        out, err = proc.communicate()
        if proc.returncode != 0:
            print(proc.returncode)
        if (out != b''):
            print(out.decode())
        if (err != b''):
            print(err.decode())
        if proc.returncode != 0:
            sys.exit()


def print_main(string):
    # Running forward simulation
    print("\n")
    print(72 * "=")
    print(f"{f' {string} ':=^72}")
    print(72 * "=")
    print("\n")


def print_section(string):
    # Running forward simulation
    print("\n")
    print(f"{f' {string} ':=^72}")
    print("\n")


def print_action(string):
    print(f"---> {string} ...")


def process_wrapper(st: Stream, event: CMTSource, paramdict: dict,
                    inv: Union[Inventory, None] = None,
                    observed: bool = True):
    """Fixes start and endtime in the dictionary

    Parameters
    ----------
    stream : Stream
        stream to be processed
    paramdict : dict
        parameterdictionary
    event : CMTSource
        event

    Returns
    -------
    dict
        processparameter dict

    """

    newdict = deepcopy(paramdict)
    rstart = newdict.pop("relative_starttime")
    rend = newdict.pop("relative_endtime")
    newdict.update({
        "starttime": event.cmt_time + rstart,
        "endtime": event.cmt_time + rend,
    })
    newdict.update({"remove_response_flag": observed})

    return process_stream(st, event_latitude=event.latitude,
                          event_longitude=event.longitude,
                          inventory=inv, **newdict)


# Main parameters
scriptdir = os.path.dirname(os.path.abspath(__file__))
station_xml = '/home/lsawade/lwsspy/invdir/station2_filtered.xml'
SPECFEM = "/scratch/gpfs/lsawade/MagicScripts/specfem3d_globe"
specfem_dict = {
    "bin": "link",
    "DATA": {
        "Par_file": "file",
    },
    "DATABASES_MPI": "link",
    "OUTPUT_FILES": "dir"
}

# %% Create inversion directory and simulation directories
invdir = '/home/lsawade/lwsspy/invdir'
if os.path.exists(invdir) is False:
    os.mkdir(invdir)

datasimdir = os.path.join(invdir, 'Data')
syntsimdir = os.path.join(invdir, 'Synt')
dsynsimdir = os.path.join(invdir, 'Dsyn')

datastations = os.path.join(datasimdir, 'DATA', 'STATIONS')
syntstations = os.path.join(syntsimdir, 'DATA', 'STATIONS')
dsynstations = os.path.join(dsynsimdir, 'DATA', 'STATIONS')
data_parfile = os.path.join(datasimdir, "DATA", "Par_file")
synt_parfile = os.path.join(syntsimdir, "DATA", "Par_file")
dsyn_parfile = os.path.join(dsynsimdir, "DATA", "Par_file")
data_cmt = os.path.join(datasimdir, "DATA", "CMTSOLUTION")
synt_cmt = os.path.join(syntsimdir, "DATA", "CMTSOLUTION")
dsyn_cmt = os.path.join(dsynsimdir, "DATA", "CMTSOLUTION")
compute_synt = True
if compute_synt:

    # Create simulation dirs
    createsimdir(SPECFEM, syntsimdir, specfem_dict=specfem_dict)
    createsimdir(SPECFEM, dsynsimdir, specfem_dict=specfem_dict)

    # Set Station file
    stationxml2STATIONS(station_xml, syntstations)
    stationxml2STATIONS(station_xml, dsynstations)

    # Set data parameters
    synt_pars = read_parfile(synt_parfile)
    dsyn_pars = read_parfile(dsyn_parfile)
    synt_pars["USE_SOURCE_DERIVATIVE"] = False
    dsyn_pars["USE_SOURCE_DERIVATIVE"] = True
    dsyn_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = 1  # For depth
    write_parfile(synt_pars, synt_parfile)
    write_parfile(dsyn_pars, dsyn_parfile)


compute_data = False
if compute_data:
    # Once data is created it can be left..
    createsimdir(SPECFEM, datasimdir, specfem_dict=specfem_dict)

    # Set Station file
    stationxml2STATIONS(station_xml, datastations)

    # Set data parameters
    data_pars = read_parfile(data_parfile)
    data_pars["USE_SOURCE_DERIVATIVE"] = False
    write_parfile(data_pars, data_parfile)


# %% Get Model CMT
CMTSOLUTION = os.path.join(invdir, "CMTSOLUTION_Italy_shallow")
cmt_goal = CMTSource.from_CMTSOLUTION_file(CMTSOLUTION)
cmt_init = deepcopy(cmt_goal)
xml_event = read_events(CMTSOLUTION)[0]

# Add 10km to initial model
cmt_init.depth_in_m += 10000.0
cmt_goal.write_CMTSOLUTION_file(data_cmt)

# %%


def generate_data(specfemdir):
    """Launches specfem for a forward simulation with the target
    CMTSOLUTION model.
    """

    cmd_list = [['mpiexec', '-n', '1', './bin/xspecfem3D']]
    # This hopefully gets around the change of directory thing
    cwdlist = [specfemdir]
    run_cmds_parallel(cmd_list, cwdlist=cwdlist)


# Running forward simulation
print_main("Starting the inversion")
if compute_data:
    print_action("Generating the data")
    generate_data(datasimdir)

rawdata = read(os.path.join(datasimdir, "OUTPUT_FILES", "*.sac"))

inv = read_inventory(station_xml)
processparams = read_yaml_file(os.path.join(scriptdir, "process.body.yml"))
# with nostdout():
data = process_wrapper(rawdata, cmt_init, processparams,
                       inv=inv, observed=False)
print(f"Data # of traces: {len(data)}")
# %% cost and gradient computation


def compute_stream_cost(data, synt):

    x = 0.0

    for tr in data:
        network, station, component = (
            tr.stats.network, tr.stats.station, tr.stats.component)
        # Get the trace sampling time
        dt = tr.stats.delta
        d = tr.data

        try:
            s = synt.select(network=network, station=station,
                            component=component)[0].data

            for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                ws = s[win.left:win.right]
                wo = d[win.left:win.right]
                x += 0.5 * np.sum(tap * (ws - wo) ** 2) * dt

        except Exception as e:
            print(f"Error at ({network}.{station}.{component}): {e}")

    return x


def compute_stream_grad_depth(data, synt, dsyn):

    x = 0.0

    for tr in data:
        network, station, component = (
            tr.stats.network, tr.stats.station, tr.stats.component)

        # Get the trace sampling time
        dt = tr.stats.delta
        d = tr.data

        try:
            s = synt.select(network=network, station=station,
                            component=component)[0].data
            dsdz = dsyn.select(network=network, station=station,
                               component=component)[0].data

            for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                wsyn = s[win.left:win.right]
                wobs = d[win.left:win.right]
                wdsdz = dsdz[win.left:win.right]
                x += np.sum((wsyn - wobs) * wdsdz * tap) * dt

        except Exception as e:
            print(f"When accessing {network}.{station}.{component}")
            print(e)

    return x


iterwindow = 0


def compute_cost_and_gradient(model):
    global iterwindow, data
    # Populate the simulation directories
    cmt = deepcopy(cmt_init)
    cmt.depth_in_m = model[0]
    cmt.write_CMTSOLUTION_file(synt_cmt)
    cmt.write_CMTSOLUTION_file(dsyn_cmt)

    # Run the simulations
    cmd_list = 2 * [['mpiexec', '-n', '1', './bin/xspecfem3D']]
    cwdlist = [syntsimdir, dsynsimdir]
    print_action("Submitting simulations")
    if compute_synt:
        run_cmds_parallel(cmd_list, cwdlist=cwdlist)
    print()

    # Get streams
    synt = read(os.path.join(syntsimdir, "OUTPUT_FILES", "*.sac"))
    dsyn = read(os.path.join(dsynsimdir, "OUTPUT_FILES", "*.sac"))
    print_action("Processing Synthetic")
    # with nostdout():
    synt = process_wrapper(synt, cmt_init, processparams,
                           inv=inv, observed=False)

    print_action("Processing Fréchet")
    # with nostdout():
    dsyn = process_wrapper(dsyn, cmt_init, processparams,
                           inv=inv, observed=False)

    # After the first forward modeling window the data
    if iterwindow == 0:
        print_action("Windowing")
        window_config = read_yaml_file(
            os.path.join(scriptdir, "window.body.yml"))
        window_on_stream(data, synt, window_config,
                         station=inv, event=xml_event)
        taper_windows(data, taper_type="tukey", alpha=0.25)
        iterwindow += 1

    cost = compute_stream_cost(data, synt)
    grad = np.array([compute_stream_grad_depth(data, synt, dsyn)])

    return cost, grad


# Define initial model that is 10km off
model = np.array([cmt_goal.depth_in_m + 10000.0])

print_section("BFGS")
# Prepare optim steepest
optim = Optimization("bfgs")
optim.compute_cost_and_gradient = compute_cost_and_gradient
optim.is_preco = False
optim.niter_max = 7
optim.stopping_criterion = 1e-8
optim.n = len(model)
optim_bfgs = optim.solve(optim, model)

plot_optimization(
    optim_bfgs, outfile="depth_inversion_misfit_reduction.pdf")
plot_model_history(optim_bfgs, labellist=['depth'],
                   outfile="depth_inversion_model_history.pdf")

ax = plot_station_xml(station_xml)
ax.add_collection(
    beach(cmt_init.tensor, xy=(cmt_init.longitude, cmt_init.latitude), width=5,
          size=100, linewidth=1.0))
plt.savefig("depth_inversion_map.pdf")


# %%

tr = data[20]