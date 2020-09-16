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
    from .statistics.fakerelation import fakerelation  # type: ignore # noqa
    from .statistics.errorellipse import errorellipse  # type: ignore # noqa

    # Math
    from .math.eigsort import eigsort # type: ignore # noqa

    # Plot
    from .plot_util.updaterc import updaterc # type: ignore # noqa
    from .plot_util.remove_ticklabels import remove_xticklabels  # type: ignore # noqa
    from .plot_util.remove_ticklabels import remove_yticklabels  # type: ignore # noqa
    from .plot_util.figcolorbar import figcolorbar  # type: ignore # noqa
