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
    from .shell.cat import cat  # noqa
    from .shell.readfile import readfile  # noqa
    from .shell.writefile import writefile  # noqa

    # Weather
    from .weather.requestweather import requestweather  # noqa
