"""
This is the first script to enable depth iterations for the
CMTSOLUTTION depth.
"""
# %% Create inversion directory

# Internal
from lwsspy import download_waveforms_cmt2storage
from lwsspy import plot_station_xml
from lwsspy import process_wrapper
from lwsspy import read_inventory
from lwsspy import CMTSource
from lwsspy import add_tapers, window_on_stream
from lwsspy import stream_cost_win
from lwsspy import stream_grad_frechet_win
from lwsspy import stream_grad_and_hess_win
from lwsspy import createsimdir
from lwsspy import read_parfile
from lwsspy import stationxml2STATIONS
from lwsspy import write_parfile
from lwsspy import Optimization
from lwsspy import plot_optimization
from lwsspy import plot_model_history
from lwsspy import updaterc
from lwsspy import run_cmds_parallel
from lwsspy import read_yaml_file
from lwsspy import print_action, print_bar, print_section

# External
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_events


updaterc()


# Main parameters
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

if os.path.exists(datadir) is False:
    download_waveforms_cmt2storage(
        CMTSOLUTION, datadir,
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

compute_synt = True
if compute_synt:
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

print_bar("Starting the inversion")

# Create dictionary for processing
rawdata = read(waveforms)
inv = read_inventory(station_xml)
print_action("Processing the observed data")
processparams = read_yaml_file(os.path.join(scriptdir, "process.body.yml"))
data = process_wrapper(rawdata, cmt_init, processparams, inv=inv)

# %% Loading and Processing the data


def compute_stream_cost(data, synt):

    x = 0.0

    for tr in data:
        network, station, location, channel, component = (
            tr.stats.network, tr.stats.station, tr.stats.location,
            tr.stats.channel, tr.stats.component)
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
            print(
                f"Error - gradient - "
                f"{network}.{station}.{location}.{channel}: {e}")

    return x


def compute_stream_grad_depth(data, synt, dsyn):

    x = 0.0

    for tr in data:
        network, station, location, channel, component = (
            tr.stats.network, tr.stats.station, tr.stats.location,
            tr.stats.channel, tr.stats.component)

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
            print(
                f"Error - gradient - "
                f"{network}.{station}.{location}.{channel}: {e}")

    return x


def compute_stream_grad_and_hess(data, synt, dsyn):

    g = 0.0
    h = 0.0
    for tr in data:
        network, station, location, channel, component = (
            tr.stats.network, tr.stats.station, tr.stats.location,
            tr.stats.channel, tr.stats.component)

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
                g += np.sum((wsyn - wobs) * wdsdz * tap) * dt
                h += np.sum(wdsdz ** 2 * tap) * dt

        except Exception as e:
            print(
                f"Error - gradient/hess - "
                f"{network}.{station}.{location}.{channel}: {e}")

    return g, h


iterwindow = 0


def compute_cost_and_gradient(model):
    global iterwindow, data
    # Populate the simulation directories
    cmt = deepcopy(cmt_init)
    cmt.depth_in_m = model[0] * 1000  # Gradient is in km!
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
        print("Data")
        print(data)
        print("Synt")
        print(synt)
        print("Inv")
        print(inv)
        window_on_stream(data, synt, window_config,
                         station=inv, event=xml_event)
        add_tapers(data, taper_type="tukey", alpha=0.25)
        iterwindow += 1

    cost = stream_cost_win(data, synt)
    grad = np.array([stream_grad_frechet_win(data, synt, dsyn)])

    return cost, grad


def compute_cost_and_gradient_hessian(model):
    global iterwindow, data
    # Populate the simulation directories
    cmt = deepcopy(cmt_init)
    cmt.depth_in_m = model[0] * 1000.0  # Gradient is in km!
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
        add_tapers(data, taper_type="tukey", alpha=0.25)
        iterwindow += 1

    cost = stream_cost_win(data, synt)
    grad, hess = stream_grad_and_hess_win(data, synt, dsyn)

    return cost, np.array([grad]), np.array([[hess]])


# %% Computing the depth range
depths = np.arange(cmt_init.depth_in_m - 10000,
                   cmt_init.depth_in_m + 11000, 1000) / 1000
cost = []
grad = []
hess = []
for _dep in depths:
    print_action(f"Computing depth: {_dep}")
    c, g, h = compute_cost_and_gradient_hessian(np.array([_dep]))
    cost.append(c)
    grad.append(g[0])
    hess.append(h[0, 0])

plt.figure(figsize=(11, 4))
ax1 = plt.subplot(1, 4, 1)
plt.plot(cost, depths)
plt.ylabel('z')
plt.xlabel('Cost')
plt.gca().invert_yaxis()
ax2 = plt.subplot(1, 4, 2, sharey=ax1)
plt.plot(grad, depths)
plt.xlabel('Gradient')
ax2.tick_params(labelleft=False)
ax3 = plt.subplot(1, 4, 3, sharey=ax1)
plt.plot(hess, depths)
plt.xlabel('Hessian')
ax3.tick_params(labelleft=False)
ax4 = plt.subplot(1, 4, 4, sharey=ax1)
plt.plot(np.array(grad)/np.array(hess), depths)
plt.xlabel('Gradient/Hessian')
ax4.tick_params(labelleft=False)

plt.subplots_adjust(hspace=0.125, wspace=0.125)
plt.savefig("DataCostGradHess.pdf")


# ax = plot_station_xml(station_xml)
# ax.add_collection(
#     beach(cmt_init.tensor, xy=(cmt_init.longitude, cmt_init.latitude),
#           width=5, size=100, linewidth=1.0))
# plt.savefig("depth_inversion_map.pdf")

# # Define initial model that is 10km off
# model = np.array([cmt_init.depth_in_m]) / 1000.0

# print_section("BFGS")
# # Prepare optim steepest
# optim = Optimization("bfgs")
# optim.compute_cost_and_gradient = compute_cost_and_gradient
# optim.is_preco = False
# optim.niter_max = 7
# optim.nls_max = 3
# optim.stopping_criterion = 1e-8
# optim.n = len(model)
# optim_bfgs = optim.solve(optim, model)

# plot_optimization(
#     optim_bfgs, outfile="depth_inversion_misfit_reduction.pdf")
# plot_model_history(optim_bfgs, labellist=['depth'],
#                    outfile="depth_inversion_model_history.pdf")

# print_section("GN")
# # Prepare optim steepest
# optim = Optimization("gn")
# optim.compute_cost_and_grad_and_hess = compute_cost_and_gradient_hessian
# optim.is_preco = False
# optim.niter_max = 7
# optim.damping = 0.0
# optim.stopping_criterion = 1e-8
# optim.n = len(model)
# optim_gn = optim.solve(optim, model)

# plot_optimization(
#     [optim_bfgs, optim_gn], outfile="depth_inversion_misfit_reduction_comp.pdf")
# plot_model_history([optim_bfgs, optim_gn], labellist=['depth'],
#                    outfile="depth_inversion_model_history_comp.pdf")
