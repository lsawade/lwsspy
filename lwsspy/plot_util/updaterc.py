import matplotlib
import os
import platform


def updaterc(rebuild=False):
    """Updates the rcParams to something generic that looks ok good out of
    the box.

    Args:

        rebuild (bool):
            Rebuilds fontcache incase it needs it.

    Last modified: Lucas Sawade, 2020.09.15 01.00 (lsawade@princeton.edu)
    """
    # if platform.system() == "Darwin":
    #     # Add Helvetica from own font dir if not available
    #     # font = _add_Helvetica()
    #     font = 'sans-serif'
    # else:
    #     font = 'LiberationSans-Regular'

    # if rebuild:
    #     matplotlib.font_manager._rebuild()

    params = {
        'font.family': "Helvetica",
        'font.size': 12,
        # 'pdf.fonttype': 3,
        'font.weight': 'normal',
        'ps.useafm': True,
        'pdf.use14corefonts': True,
        'axes.unicode_minus': False,
        'axes.labelweight': 'normal',
        'axes.labelsize': 'small',
        'axes.titlesize': 'medium',
        'axes.linewidth': 1,
        'axes.grid': False,
        'grid.color': "k",
        'grid.linestyle': ":",
        'grid.alpha': 0.7,
        'xtick.labelsize': 'small',
        'xtick.direction': 'out',
        'xtick.top': True,  # draw label on the top
        'xtick.bottom': True,  # draw label on the bottom
        'xtick.minor.visible': True,
        'xtick.major.top': True,  # draw x axis top major ticks
        'xtick.major.bottom': True,  # draw x axis bottom major ticks
        'xtick.major.size': 4,  # draw x axis top major ticks
        'xtick.major.width': 1,  # draw x axis top major ticks
        'xtick.minor.top': True,  # draw x axis top minor ticks
        'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'xtick.minor.width': 1,  # draw x axis top major ticks
        'xtick.minor.size': 2,  # draw x axis top major ticks
        'ytick.labelsize': 'small',
        'ytick.direction': 'out',
        'ytick.left': True,  # draw label on the top
        'ytick.right': True,  # draw label on the bottom
        'ytick.minor.visible': True,
        'ytick.major.left': True,  # draw x axis top major ticks
        'ytick.major.right': True,  # draw x axis bottom major ticks
        'ytick.major.size': 4,  # draw x axis top major ticks
        'ytick.major.width': 1,  # draw x axis top major ticks
        'ytick.minor.left': True,  # draw x axis top minor ticks
        'ytick.minor.right': True,  # draw x axis bottom minor ticks
        'ytick.minor.size': 2,  # draw x axis top major ticks
        'ytick.minor.width': 1,  # draw x axis top major ticks
        'legend.fancybox': False,
        'legend.frameon': True,
        'legend.loc': 'best',
        'legend.numpoints': 1,
        'legend.fontsize': 'small',
        'legend.framealpha': 1,
        'legend.scatterpoints': 3,
        'legend.edgecolor': 'inherit',
        'legend.facecolor': 'w'
        # 'mathtext.fontset': 'custom',
        # 'mathtext.rm': 'Bitstream Vera Sans',
        # 'mathtext.it': 'Bitstream Vera Sans:italic',
        # 'mathtext.bf':'Bitstream Vera Sans:bold'
    }
    matplotlib.rcParams.update(params)


# def _add_Helvetica():

#     # Check if Helvetica in system fonts
#     from matplotlib import font_manager
#     fonts = [os.path.basename(x).split(".")[0]
#              for x in font_manager.findSystemFonts(
#         fontpaths=None)]
#     fonts.sort()
#     # print(fonts)
#     if "HelveticaNeue" in fonts:
#         pass
#     elif "Helvetica Neue" in fonts:
#         pass
#     elif "Helvetica" in fonts:
#         return "Helvetica"
#     else:
#         font_file = os.path.join(
#             os.path.dirname(__file__), 'fonts', 'HelveticaNeue.ttc')
#         font_manager.fontManager.addfont(font_file)
#     return "Helvetica Neue"


def updaterc_pres(rebuild=False):
    """Updates the rcParams to something generic that looks ok good out of
    the box.

    Args:

        rebuild (bool):
            Rebuilds fontcache incase it needs it.

    Last modified: Lucas Sawade, 2020.09.15 01.00 (lsawade@princeton.edu)
    """

    params = {
        'font.family': 'Helvetica Neue',
        'pdf.fonttype': 42,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'axes.linewidth': 2,
        'axes.grid': True,
        'grid.color': "lightgrey",
        'xtick.labelsize': 12,
        'xtick.direction': 'inout',
        'xtick.top': True,  # draw label on the top
        'xtick.bottom': True,  # draw label on the bottom
        'xtick.minor.visible': True,
        'xtick.major.top': True,  # draw x axis top major ticks
        'xtick.major.bottom': True,  # draw x axis bottom major ticks
        'xtick.major.size': 8,  # draw x axis top major ticks
        'xtick.major.width': 2,  # draw x axis top major ticks
        'xtick.minor.top': True,  # draw x axis top minor ticks
        'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'xtick.minor.width': 2,  # draw x axis top major ticks
        'xtick.minor.size': 5,  # draw x axis top major ticks
        'ytick.labelsize': 12,
        'ytick.direction': 'inout',
        'ytick.left': True,  # draw label on the top
        'ytick.right': True,  # draw label on the bottom
        'ytick.minor.visible': True,
        'ytick.major.left': True,  # draw x axis top major ticks
        'ytick.major.right': True,  # draw x axis bottom major ticks
        'ytick.major.size': 8,  # draw x axis top major ticks
        'ytick.major.width': 2,  # draw x axis top major ticks
        'ytick.minor.left': True,  # draw x axis top minor ticks
        'ytick.minor.right': True,  # draw x axis bottom minor ticks
        'ytick.minor.size': 5,  # draw x axis top major ticks
        'ytick.minor.width': 2,  # draw x axis top major ticks
        'legend.fancybox': False,
        'legend.frameon': True,
        'legend.loc': 'best',
        'legend.numpoints': 1,
        'legend.fontsize': 12,
        'legend.framealpha': 1,
        'legend.scatterpoints': 3,
        'legend.edgecolor': 'inherit',
        'legend.facecolor': 'w',
        'mathtext.fontset': 'stix',
        # 'mathtext.rm': 'Helvetica',
        # 'mathtext.it': 'Helvetica:italic',
        # 'mathtext.bf': 'Helvetica:bold'
    }
    matplotlib.rcParams.update(params)

    if rebuild:
        matplotlib.font_manager._rebuild()
