from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import lwsspy as lpy
import _pickle as cPickle


def plot_measurement_pkl(measurement_pickle: str):

    with open(measurement_pickle, "rb") as f:
        measurements = cPickle.load(f)

    plot_measurements(measurements)


def plot_measurements(measurements: dict):

    # Get number of wave types:
    Nwaves = len(measurements.keys())

    # Get the amount of colors
    colors = lpy.pick_colors_from_cmap(Nwaves*3, cmap='rainbow')

    # Create base figure
    fig = plt.figure(figsize=(8, 0.5+Nwaves*1.5))
    gs = GridSpec(Nwaves, 3, figure=fig)
    # plt.subplots_adjust(bottom=0.075, top=0.95,
    #                     left=0.05, right=0.95, hspace=0.25)
    plt.subplots_adjust(bottom=0.2, top=0.9,
                        left=0.1, right=0.9, hspace=0.3)

    # Create subplots
    counter = 0
    components = ["Z", "R", "T"]
    component_bins = [50, 20, 10]

    for _i, (_wtype, _compdict) in enumerate(measurements.items()):
        for _j, (_comp, _bins) in enumerate(zip(components, component_bins)):

            # Get the data type from the measurement dictionary
            _residuals = np.array(_compdict[_comp]["dL2"])
            _norm2 = np.array(_compdict[_comp]["L2"])
            _trace_nrj = np.array(_compdict[_comp]["trace_energy"])

            # Set alpha color
            acolor = deepcopy(colors[counter, :])
            acolor[3] = 0.5

            # Create plot
            ax = plt.subplot(gs[_i, _j])
            plt.hist(_residuals/_norm2,
                     bins=_bins,
                     edgecolor=colors[counter, :],
                     facecolor='none', linewidth=0.75,
                     label='GCMT', histtype='step')
            # lpy.plot_label(ax, lpy.abc[counter] + ")", location=6, box=False,
            #                fontsize="small")
            lpy.plot_label(ax, f"N: {len(_residuals)}", location=2, box=False,
                           fontsize="small")
            # if _wtype == "body" and _comp == "Z":
            #     ax.set_xlim((-0.001, 0.1))

            if _j == 0:
                plt.ylabel(_wtype.capitalize())

            if _i == Nwaves-1:
                plt.xlabel(_comp.capitalize())
            else:
                pass
                # ax.tick_params(labelbottom=False)

            counter += 1

    plt.show()
