import sys
import os

# Global imports when running the modules for testing
if "-m" in sys.argv:
    DOCFIGURES = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))),
        'docs', 'source', 'chapters', 'figures')
else:
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
