import matplotlib.font_manager as fm
import matplotlib
import os
import glob
import platform
import matplotlib.ft2font as ft

FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')


def updaterc(rebuild=False):
    """Updates the rcParams to something generic that looks ok good out of
    the box.

    Args:

        rebuild (bool):
            Rebuilds fontcache incase it needs it.

    Last modified: Lucas Sawade, 2020.09.15 01.00 (lsawade@princeton.edu)
    """

    add_fonts()

    params = {
        'font.family': 'sans-serif',
        'font.style':   'normal',
        'font.variant': 'normal',
        'font.weight':  'normal',
        'font.stretch': 'normal',
        'font.size':    12.0,
        'font.serif':     [
            'Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman',
            'New Century Schoolbook', 'Century Schoolbook L', 'Utopia',
            'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L',
            'Times', 'Palatino', 'Charter', 'serif'
        ],
        'font.sans-serif': [
            'Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans',
            'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana',
            'Geneva', 'Lucid', 'Avant Garde', 'sans-serif'
        ],
        'font.cursive':    [
            'Apple Chancery', 'Textile', 'Zapf Chancery', 'Sand', 'Script MT',
            'Felipa', 'Comic Neue', 'Comic Sans MS', 'cursive'
        ],
        'font.fantasy':    [
            'Chicago', 'Charcoal', 'Impact', 'Western', 'Humor Sans', 'xkcd',
            'fantasy'
        ],
        'font.monospace':  [
            'Roboto Mono', 'Monaco', 'DejaVu Sans Mono',
            'Bitstream Vera Sans Mono',  'Computer Modern Typewriter',
            'Andale Mono', 'Nimbus Mono L', 'Courier New', 'Courier', 'Fixed',
            'Terminal', 'monospace'
        ],
        'font.size': 12,
        # 'pdf.fonttype': 3,
        'font.weight': 'normal',
        # 'pdf.fonttype': 42,
        # 'ps.fonttype': 42,
        # 'ps.useafm': True,
        # 'pdf.use14corefonts': True,
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
        'legend.facecolor': 'w',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'sans',
        'mathtext.it': 'sans:italic',
        'mathtext.bf': 'sans:bold',
        'mathtext.cal': 'cursive',
        'mathtext.tt':  'monospace',
        'mathtext.fallback': 'cm',
        'mathtext.default': 'it'
    }

    matplotlib.rcParams.update(params)


def add_fonts(verbose: bool = False):

    # Remove fontlist:
    for file in glob.glob('~/.matplotlib/font*.json'):
        os.remove(file)

    # Fonts
    fontfiles = glob.glob(os.path.join(FONTS, "*.tt?"))

    # for name, fname in fontdict.items():
    for fname in fontfiles:

        font = ft.FT2Font(fname)

        # Just to verify what kind of fonts are added verifiably
        if verbose:
            print(fname, "Scalable:", font.scalable)
            for style in ('Italic',
                          'Bold',
                          'Scalable',
                          'Fixed sizes',
                          'Fixed width',
                          'SFNT',
                          'Horizontal',
                          'Vertical',
                          'Kerning',
                          'Fast glyphs',
                          'Multiple masters',
                          'Glyph names',
                          'External stream'):
                bitpos = getattr(ft, style.replace(' ', '_').upper()) - 1
                print(f"{style+':':17}", bool(font.style_flags & (1 << bitpos)))

        # Actually adding the fonts
        fe = fm.ttfFontProperty(font)
        fm.fontManager.ttflist.insert(0, fe)

    # matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
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
