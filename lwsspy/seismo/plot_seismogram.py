
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from obspy import Trace
from .source import CMTSource
from ..plot_util.plot_label import plot_label
from ..math.envelope import envelope


def plot_seismogram(obsd: Trace,
                    synt: Optional[Trace] = None,
                    cmtsource: Optional[CMTSource] = None,
                    tag: Optional[str] = None):
    station = obsd.stats.station
    network = obsd.stats.network
    channel = obsd.stats.channel
    location = obsd.stats.location

    trace_id = f"{network}.{station}.{location}.{channel}"

    # Times and offsets computed individually, since the grid search applies
    # a timeshift which changes the times of the traces.
    if cmtsource is None:
        offset = 0
        if isinstance(synt, Trace):
            offset_synt = 0
    else:
        offset = obsd.stats.starttime - cmtsource.cmt_time
        if isinstance(synt, Trace):
            offset_synt = synt.stats.starttime - cmtsource.cmt_time

    times = [offset + obsd.stats.delta * i for i in range(obsd.stats.npts)]
    if isinstance(synt, Trace):
        times_synt = [offset_synt + synt.stats.delta * i
                      for i in range(synt.stats.npts)]

    # Figure Setup
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(211)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95)

    ax1.plot(times, obsd.data, color="black", linewidth=0.75,
             label="Observed")
    if isinstance(synt, Trace):
        ax1.plot(times_synt, synt.data, color="red", linewidth=0.75,
                 label="Synthetic")
    ax1.set_xlim(times[0], times[-1])
    ax1.legend(loc='upper right', frameon=False, ncol=3, prop={'size': 11})
    ax1.tick_params(labelbottom=False, labeltop=False)

    # Setting top left corner text manually
    if isinstance(tag, str):
        label = f"{trace_id}\n{tag.capitalize()}"
    else:
        label = f"{trace_id}"
    plot_label(ax1, label, location=1, dist=0.005, box=False)

    # plot envelope
    ax2 = plt.subplot(212)
    ax2.plot(times, envelope(obsd.data), color="black",
             linewidth=1.0, label="Observed")
    if isinstance(synt, Trace):
        ax2.plot(times_synt, envelope(synt.data), color="red", linewidth=1,
                 label="Synthetic")
    ax2.set_xlim(times[0], times[-1])
    ax2.set_xlabel("Time [s]", fontsize=13)
    plot_label(ax2, "Envelope", location=1, dist=0.005, box=False)
    if isinstance(synt, Trace):
        try:
            for win in obsd.stats.windows:
                left = times[win.left]
                right = times[win.right]
                re1 = Rectangle((left, ax1.get_ylim()[0]), right - left,
                                ax1.get_ylim()[1] - ax1.get_ylim()[0],
                                color="blue", alpha=0.25, zorder=-1)
                ax1.add_patch(re1)
                re2 = Rectangle((left, ax2.get_ylim()[0]), right - left,
                                ax2.get_ylim()[1] - ax2.get_ylim()[0],
                                color="blue", alpha=0.25, zorder=-1)
                ax2.add_patch(re2)
        except Exception as e:
            print(e)

    return fig
