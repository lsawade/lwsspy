import sys

# Global imports when running the modules for testing
if "-m" in sys.argv:
    pass
else:
    # Statistics
    from .statistics.fakerelation import fakerelation  # type: ignore # noqa
    from .statistics.errorellipse import errorellipse  # type: ignore # noqa

    # Math
    from .math.eigsort import eigsort # type: ignore # noqa

    # Plot
    from .plot.updaterc import updaterc # type: ignore # noqa