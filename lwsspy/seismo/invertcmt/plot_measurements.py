import _pickle as cPickle
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from glob import glob
from typing import Optional
import os
import lwsspy as lpy


def get_bins(b, a, nbin, mtype):

    if mtype == "max_cc":
        ax_min = np.min((np.min(b), np.min(a)))
        ax_max = np.max((np.max(b), np.max(a)))
    elif mtype == "chi":
        ax_min = 0.0
        ax_max = 2.0  # np.max((np.max(b), np.max(a)))
    elif mtype == "misfit":
        ax_min = 0.0
        ax_max = 2.0  # np.max((np.max(b), np.max(a)))
    elif mtype == "time_shift":
        ax_min = -20.0
        ax_max = 20.0  # np.max((np.max(b), np.max(a)))
    else:
        ax_min = np.min((np.min(b), np.min(a)))
        ax_max = np.max((np.max(b), np.max(a)))
        abs_max = np.max((np.abs(ax_min), np.abs(ax_max)))
        ax_min = -abs_max
        ax_max = abs_max
    binwidth = (ax_max - ax_min) / nbin

    return np.arange(ax_min, ax_max + binwidth / 2., binwidth)


def plot_measurement_pkl(
        measurement_pickle_before: str,
        measurement_pickle_after: str):

    with open(measurement_pickle_before, "rb") as f:
        measurements_before = cPickle.load(f)
    with open(measurement_pickle_after, "rb") as f:
        measurements_after = cPickle.load(f)

    plot_measurements(measurements_before, measurements_after)


def get_measurement(bdict: dict, adict: dict, mtype: str):

    if mtype == "chi":
        # Get the data type from the measurement dictionary
        b = np.array(bdict["dL2"])/np.array(bdict["L2"])
        a = np.array(adict["dL2"])/np.array(adict["L2"])
    elif mtype == "misfit":
        # Get the data type from the measurement dictionary
        b = np.array(bdict["dL2"])/np.array(bdict["trace_energy"])
        a = np.array(adict["dL2"])/np.array(adict["trace_energy"])

    elif mtype == "misfit":
        # Get the data type from the measurement dictionary
        b = np.array(bdict["dlna"])
        a = np.array(adict["dlna"])
    else:
        b = np.array(bdict[mtype])
        a = np.array(adict[mtype])

    return b, a


def plot_measurements(before: dict, after: dict, alabel: Optional[str] = None,
                      blabel: Optional[str] = None, mtype='chi'):

    # Get number of wave types:
    Nwaves = len(before.keys())

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
    if mtype == "time_shift":
        component_bins = [21, 21, 21]
    else:
        component_bins = [75, 75, 75]

    if blabel is None:
        blabel = "$m_0$"

    if alabel is None:
        alabel = "$m_f$"

    for _i, (_wtype, _compdict) in enumerate(before.items()):
        for _j, (_comp, _bins) in enumerate(zip(components, component_bins)):

            bdict = _compdict[_comp]
            adict = after[_wtype][_comp]

            b, a = get_measurement(bdict, adict, mtype)
            # Set alpha color
            acolor = deepcopy(colors[counter, :])
            acolor[3] = 0.5

            # Create plot
            ax = plt.subplot(gs[_i, _j])

            # Plot before
            bins = get_bins(b, a, _bins, mtype)
            nb, _, _ = plt.hist(b,
                                bins=bins,
                                edgecolor=colors[counter, :],
                                facecolor='none', linewidth=0.75, linestyle="--",
                                histtype='step')
            plt.plot([], [], color=colors[counter, :],
                     linewidth=0.75, linestyle=":", label=blabel)

            # Plot After
            na, _, _ = plt.hist(a,
                                bins=bins,
                                edgecolor=colors[counter, :],
                                facecolor='none', linewidth=1.0, linestyle="-",
                                histtype='step')
            plt.plot([], [], color=colors[counter, :],
                     linewidth=0.75, linestyle="-", label=alabel)

            # Annotations
            lpy.plot_label(ax, f"N: {len(b)}", location=2, box=False,
                           fontsize="small")
            ax.set_ylim((0, 1.275*np.max([np.max(nb), np.max(na)])))
            plt.legend(loc='upper left', fontsize='x-small',
                       fancybox=False, frameon=False,
                       ncol=2, borderaxespad=0.0, borderpad=0.5,
                       handletextpad=0.15, labelspacing=0.0,
                       handlelength=1.0, columnspacing=1.0)
            # lpy.plot_label(ax, lpy.abc[counter] + ")", location=6, box=False,
            #                fontsize="small")

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


def get_database_measurements(database: str, outdir: Optional[str] = None):

    # Get all directories
    cmtlocs = glob(os.path.join(database, '*'))

    # Empty measurement lists
    components = ["Z", "R", "T"]
    for _cmtloc in cmtlocs:
        print(_cmtloc)
        try:
            measurement_pickle_before = os.path.join(
                _cmtloc, "measurements_before.pkl"
            )
            measurement_pickle_after = os.path.join(
                _cmtloc, "measurements_after.pkl"
            )
            print(measurement_pickle_before)
            print(measurement_pickle_after)
            with open(measurement_pickle_before, "rb") as f:
                measurements_before = cPickle.load(f)
            with open(measurement_pickle_after, "rb") as f:
                measurements_after = cPickle.load(f)

        except Exception as e:
            print(e)
            continue

        if "after" not in locals():
            before = measurements_before
            after = measurements_after

        else:

            for _wtype in measurements_before.keys():
                for _comp in components:
                    for _mtype in before[_wtype][_comp].keys():

                        # Grab
                        bdict = measurements_before[_wtype][_comp]
                        adict = measurements_after[_wtype][_comp]

                        # Get measurements
                        b, a = get_measurement(bdict, adict, _mtype)

                        # Add to first dictionary
                        before[_wtype][_comp][_mtype].extend(b)
                        after[_wtype][_comp][_mtype].extend(a)

    if outdir is not None:

        measurement_pickle_before_out = os.path.join(
            outdir, "database_measurement_before.pkl"
        )
        measurement_pickle_after_out = os.path.join(
            outdir, "database_measurement_before.pkl"
        )

        with open(measurement_pickle_before_out, "wb") as f:
            cPickle.dump(before, f)

        with open(measurement_pickle_after_out, "wb") as f:
            cPickle.dump(after, f)
    return before, after


def bin():

    import argparse
    import lwsspy as lpy

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--database', dest='database',
                        help='Database directory',
                        type=str, required=True)
    parser.add_argument('-o', '--outdir', dest='outdir',
                        help='Plot output directory',
                        required=True, type=str)
    parser.add_argument('-m', '--measurement', dest='measure', nargs='+',
                        type=str, default='chi')
    parser.add_argument('-a', '--alabel', dest='alabel',
                        type=str, default=None)
    parser.add_argument('-b', '--blabel', dest='blabel',
                        type=str, default=None)

    args = parser.parse_args()

    # Get the measurements
    before, after = lpy.get_database_measurements("testdatabase_mt_stats")

    if type(args.measure) is str:
        measure = [args.measure]
    else:
        measure = args.measure

    # Plot the measurements
    for _m in measure:
        lpy.plot_measurements(before, after, args.alabel,
                              args.blabel, mtype=_m)

        if args.outdir is not None:
            outfile = os.path.join(args.outdir, f"histograms_{_m}.pdf")
            plt.savefig(outfile, format='pdf')
