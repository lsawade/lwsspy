"""
This is the first script to enable depth iterations for the
CMTSOLUTTION depth.
"""
# %% Create inversion directory

# External
import os
import sys
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_events
from obspy.imaging.beachball import beach

# Internal
from lwsspy import plot_station_xml
from lwsspy import process_wrapper
from lwsspy import read_inventory
from lwsspy import CMTSource
from lwsspy import window_on_stream
from lwsspy import add_tapers
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
updaterc()


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

compute_data = True
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
shallow = True
if shallow:
    eventype = "Shallow"
    cmtfile = "CMTSOLUTION_Italy_shallow"
else:
    eventype = "Deep"
    cmtfile = "CMTSOLUTION"

CMTSOLUTION = os.path.join(invdir, cmtfile)
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
print_bar("Starting the inversion")

if compute_data:
    print_action("Generating the data")
    generate_data(datasimdir)

# sys.exit()

# Loading Station Data
inv = read_inventory(station_xml)

# Loading Seismic Data
rawdata = read(os.path.join(datasimdir, "OUTPUT_FILES", "*.sac"))

# Loading Process Parameters
processparams = read_yaml_file(os.path.join(scriptdir, "process.body.yml"))

# Processing observed data
data = process_wrapper(rawdata, cmt_init, processparams,
                       inv=inv, observed=False)

# Checking how many Traces are left
print(f"Data # of traces: {len(data)}")


# %% cost and gradient computation
iterwindow = 0

# Define function that iterates over depth


def compute_cost_and_gradient(model):
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


depths = np.arange(cmt_goal.depth_in_m - 10000,
                   cmt_goal.depth_in_m + 11000, 1000) / 1000.0
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
plt.savefig(f"SyntheticCostGradHess{eventype}.pdf")


# # Define initial model that is 10km off
# model = np.array([cmt_goal.depth_in_m + 10000.0])/1000.0

# print_section("BFGS")
# # Prepare optim steepest
# optim = Optimization("bfgs")
# optim.compute_cost_and_gradient = compute_cost_and_gradient
# optim.is_preco = False
# optim.niter_max = 7
# optim.stopping_criterion = 1e-8
# optim.n = len(model)
# optim_bfgs = optim.solve(optim, model)

# # plot_optimization(
# #     optim_bfgs, outfile=f"SyntheticDepthInversionMisfitReduction{eventype}.pdf")
# # plot_model_history(optim_bfgs, labellist=['depth'],
# #                    outfile=f"SyntheticDepthInversionModelHistory{eventype}.pdf")

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
#     [optim_bfgs, optim_gn],
#     outfile=f"SyntheticDepthInversionMisfitReduction{eventype}.pdf")
# plot_model_history([optim_bfgs, optim_gn], labellist=['depth'],
#                    outfile=f"SyntheticDepthInversionModelHistory{eventype}.pdf")

# ax = plot_station_xml(station_xml)
# ax.add_collection(
#     beach(cmt_init.tensor, xy=(cmt_init.longitude, cmt_init.latitude),
#           width=5, size=100, linewidth=1.0))
# plt.savefig("SyntheticAcquisition.pdf")
