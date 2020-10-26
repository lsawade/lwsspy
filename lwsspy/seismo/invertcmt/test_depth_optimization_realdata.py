"""
This is the first script to enable depth iterations for the
CMTSOLUTTION depth.
"""
# %% Create inversion directory

# External
import contextlib
import os
import sys
from typing import Union
from copy import deepcopy
from subprocess import Popen, PIPE
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_events, Stream, Trace, Inventory
import pyflex

# Internal
from lwsspy import CMTSource
from lwsspy import createsimdir
from lwsspy import download_waveforms_cmt2storage
from lwsspy import Optimization
from lwsspy import process_stream
from lwsspy import read_parfile
from lwsspy import read_yaml_file
from lwsspy import read_inventory
from lwsspy import stationxml2STATIONS
from lwsspy import updaterc
from lwsspy import write_parfile
from lwsspy import nostdout

from lwsspy.inversion.plot_optimization import plot_optimization
updaterc()


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


def print_bar(string):
    # Running forward simulation
    print("\n\n")
    print(72 * "=")
    print(f"{f' {string} ':=^72}")
    print(72 * "=")
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
        "starttime": event.origin_time + rstart,
        "endtime": event.origin_time + rend,
    })
    newdict.update({"remove_response_flag": observed})

    return process_stream(st, event_latitude=event.latitude,
                          event_longitude=event.longitude,
                          inventory=inv, **newdict)


# Main parameters
# station_xml = '/home/lsawade/lwsspy/invdir_real/station2_filtered.xml'
observed_data = ''
window_dict = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "body.window.yml")

SPECFEM = "/scratch/gpfs/lsawade/MagicScripts/specfem3d_globe"
specfem_dict = {
    "bin": "link",
    "DATA": {
        "Par_file": "file",
    },
    "DATABASES_MPI": "link",
    "OUTPUT_FILES": "dir"
}
invdir = '/home/lsawade/lwsspy/invdir_real'
datadir = os.path.join(invdir, "Data")
scriptdir = os.path.dirname(os.path.abspath(__file__))

# %% Get Model CMT

CMTSOLUTION = os.path.join(invdir, "CMTSOLUTION")
cmt_init = CMTSource.from_CMTSOLUTION_file(CMTSOLUTION)
xml_event = read_events(CMTSOLUTION)[0]


# %% Create inversion directory and simulation directories
if os.path.exists(invdir) is False:
    os.mkdir(invdir)

    download_waveforms_cmt2storage(
        CMTSOLUTION, "invdir_real/Data",
        duration=1200, stationxml="invdir_real/station2_filtered.xml")

station_xml = os.path.join(datadir, "stations", "*.xml")
waveforms = os.path.join(datadir, "waveforms", "*.mseed")

# %%
# Creating forward simulation directories for the synthetic
# and the derivate simulations
syntsimdir = os.path.join(invdir, 'Synt')
dsynsimdir = os.path.join(invdir, 'Dsyn')
syntstations = os.path.join(syntsimdir, 'DATA', 'STATIONS')
dsynstations = os.path.join(dsynsimdir, 'DATA', 'STATIONS')
synt_parfile = os.path.join(syntsimdir, "DATA", "Par_file")
dsyn_parfile = os.path.join(dsynsimdir, "DATA", "Par_file")
synt_cmt = os.path.join(syntsimdir, "DATA", "CMTSOLUTION")
dsyn_cmt = os.path.join(dsynsimdir, "DATA", "CMTSOLUTION")
createsimdir(SPECFEM, syntsimdir, specfem_dict=specfem_dict)
createsimdir(SPECFEM, dsynsimdir, specfem_dict=specfem_dict)

#  Fix Stations!
stationxml2STATIONS(station_xml, syntstations)
stationxml2STATIONS(station_xml, dsynstations)

# Get Par_file Dictionaries
synt_pars = read_parfile(synt_parfile)
dsyn_pars = read_parfile(dsyn_parfile)

# Set data parameters and  write new parfiles
synt_pars["USE_SOURCE_DERIVATIVE"] = False
dsyn_pars["USE_SOURCE_DERIVATIVE"] = True
dsyn_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = 1  # For depth
write_parfile(synt_pars, synt_parfile)
write_parfile(dsyn_pars, dsyn_parfile)


# Create dictionary for processing
rawdata = read(waveforms)
print(rawdata)
inv = read_inventory(station_xml)
print(inv)
print_action("Processing the observed data")
processparams = read_yaml_file(os.path.join(scriptdir, "process.body.yml"))
with nostdout():
    data = process_wrapper(rawdata, cmt_init, processparams, inv=inv)
print(data)
sys.exit()
# %% Fix Generate Data
print_bar("Starting the inversion")

# %% Loading and Processing the data


def compute_stream_cost(synt, data):

    x = 0.0

    for tr in synt:
        network, station, component = (
            tr.stats.network, tr.stats.station, tr.stats.component)
        # Get the trace sampling time
        dt = tr.stats.delta
        s = tr.data

        try:
            d = data.select(network=network, station=station,
                            component=component)[0].data
            x += 0.5 * np.sum((s - d) ** 2) * dt
        except Exception as e:
            print(f"When accessing {network}.{station}.{component}")
            print(e)

    return x


def compute_stream_grad_depth(synt, data, dsyn):

    x = 0.0

    for tr in synt:
        network, station, component = (
            tr.stats.network, tr.stats.station, tr.stats.component)

        # Get the trace sampling time
        dt = tr.stats.delta
        s = tr.data

        try:
            d = data.select(network=network, station=station,
                            component=component)[0].data
            dsdz = dsyn.select(network=network, station=station,
                               component=component)[0].data
            x += np.sum((s - d) * dsdz) * dt

        except Exception as e:
            print(f"When accessing {network}.{station}.{component}")
            print(e)

    return x


def compute_cost_and_gradient(model):

    # Populate the simulation directories
    cmt_init.depth_in_m = model[0]
    cmt_init.write_CMTSOLUTION_file(synt_cmt)
    cmt_init.write_CMTSOLUTION_file(dsyn_cmt)

    # Run the simulations
    cmd_list = 2 * [['mpiexec', '-n', '1', './bin/xspecfem3D']]
    cwdlist = [syntsimdir, dsynsimdir]
    print_action("Submitting simulations")
    run_cmds_parallel(cmd_list, cwdlist=cwdlist)
    print()

    # Get streams
    synt = read(os.path.join(syntsimdir, "OUTPUT_FILES", "*.sac"))
    dsyn = read(os.path.join(dsynsimdir, "OUTPUT_FILES", "*.sac"))
    print_action("Processing synthetic")
    with nostdout():
        synt = process_wrapper(synt, cmt_init, processparams,
                               inv=inv, observed=False)

    print_action("Processing Fréchet")
    with nostdout():
        dsyn = process_wrapper(dsyn, cmt_init, processparams,
                               inv=inv, observed=False)

    # print(f"Check number of traces: S: {len(synt)}, dS: {len(dsyn)}")

    cost = compute_stream_cost(synt, data)
    grad = np.array([compute_stream_grad_depth(synt, data, dsyn)])

    return cost, grad


# Define initial model that is 10km off
model = np.array([cmt_goal.depth_in_m + 10000.0])


print(f"{' BFGS ':*^72}")
# Prepare optim steepest
optim = Optimization("bfgs")
optim.compute_cost_and_gradient = compute_cost_and_gradient
optim.is_preco = False
optim.niter_max = 50
optim.stopping_criterion = 1e-8
optim.n = len(model)
optim_bfgs = optim.solve(optim, model)

plot_optimization(optim_bfgs, outfile="depth_inversion.pdf")
