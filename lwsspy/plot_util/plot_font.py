import matplotlib.pyplot as plt
import sys


def plot_font(fontname: str):
    """Plots the set of weights and styles for a specific font

    Parameters
    ----------
    fontname : str
        Name of the font to be plotted
    """

    # Create weight ditionatry for font
    weights = {
        fontname: ['regular', 'medium', 'bold'],
    }

    styles = ['normal', 'italic', 'oblique']

    alignment = {'horizontalalignment': 'center',
                 'verticalalignment': 'baseline'}

    combinations = []
    for family in weights.keys():
        for style in styles:
            for weight in weights[family]:
                combinations.append((family, weight, style))
    N = len(combinations)

    def textPlot(ax, i, N, family, weight, style):
        y = 1.-(1./float(N)) - float(i)/(float(N)+1)
        ax.text(0.5, y, family+' '+weight+' '+style,
                family=family, weight=weight, style=style,
                fontsize=30, **alignment)

    fig = plt.figure(figsize=(8, .7*N), frameon=False)
    ax = plt.gca()
    ax.axis('off')
    plt.xlim((0., 1.))
    plt.ylim((0., 1.))

    for i, c in enumerate(combinations):
        textPlot(ax, i, N, c[0], c[1], c[2])
    plt.tight_layout()
    plt.show(block=True)


def bin():

    fontname = sys.argv[1]

    plot_font(fontname)
