from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import lwsspy as lpy


def plot_residuals(residuals: dict):

    # Get number of wave types:
    Nwaves = len(residuals.keys())

    # Get the amount of colors
    colors = lpy.pick_colors_from_cmap(Nwaves*3, cmap='rainbow')
    # Create base figure
    fig = plt.figure(figsize=(10, 1+Nwaves*2))
    gs = GridSpec(Nwaves, 3, figure=fig)
    plt.subplots_adjust(bottom=0.075, top=0.95,
                        left=0.05, right=0.95, hspace=0.25)

    # Create subplots
    counter = 0
    for _i, (_wtype, _compdict) in enumerate(residuals.items()):
        for _j, (_comp, _residuals) in enumerate(_compdict.items()):

            # Set alpha color
            acolor = deepcopy(colors[counter, :])
            acolor[3] = 0.8

            # Create plot
            ax = plt.subplot(gs[_i, _j])
            plt.hist(_residuals, bins=25, edgecolor=colors[counter, :],
                     facecolor=acolor, linewidth=0.75,
                     label='GCMT', histtype='stepfilled')
            lpy.plot_label(ax, lpy.abc[counter] + ")", location=6, box=False)

            if _j == 0:
                plt.ylabel(_wtype.capitalize())

            if _i == Nwaves-1:
                plt.xlabel(_comp.capitalize())
            else:
                ax.tick_params(labelbottom=False)

            counter += 1

    plt.show()
