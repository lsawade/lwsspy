import sys
import os

DOCFIGURES = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))),
        'docs', 'source', 'chapters', 'figures')

# Global imports when running the modules for testing
if "-m" not in sys.argv:
    # Statistics
    from .statistics.fakerelation import fakerelation  # noqa
    from .statistics.errorellipse import errorellipse  # noqa

    # Math
    from .math.eigsort import eigsort  # noqa

    # Plot
    from .plot_util.updaterc import updaterc  # noqa
    from .plot_util.remove_ticklabels import remove_xticklabels  # noqa
    from .plot_util.remove_ticklabels import remove_yticklabels  # noqa
    from .plot_util.figcolorbar import figcolorbar  # noqa

    # Shell
    from .shell.ln import ln  # noqa
    from .shell.cp import cp  # noqa
    from .shell.cat import cat  # noqa
    from .shell.cpdir import cpdir  # noqa
    from .shell.touch import touch  # noqa
    from .shell.readfile import readfile  # noqa
    from .shell.writefile import writefile  # noqa
    from .shell.copy_dirtree import copy_dirtree  # noqa
    from .shell.create_dirtree import create_dirtree  # noqa

    # Weather
    from .weather.requestweather import requestweather  # noqa
    from .weather.weather import weather  # noqa
    from .weather.drop2pickle import drop2pickle  # noqa

    # Seismology
    from .seismo.source import CMTSource # noqa
    from .seismo.perturb_cmt import perturb_cmt # noqa
    from .seismo.perturb_cmt import perturb_cmt_dir # noqa
    from .seismo.validate_cmt import validate_cmt # noqa
    from .seismo.specfem.cmt2simdir import cmt2simdir  # noqa
    from .seismo.specfem.cmtdir2simdirs import cmt2simdir  # noqa
    from .seismo.specfem.createsimdir import createsimdir  # noqa


