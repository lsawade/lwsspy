import logging

from logging import getLoggerClass
from logging import NOTSET


class CustomFormatter(logging.Formatter):
    """
    Logging Formatter to add colors and count warning / errors

    This class organizes the customization of the logging output.
    The formatter as of now outputs the logs in the following manner in
    order of Loglevel:

    .. rubric:: Example Output

    .. code-block:: python

        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- INFO    : Initializing matrices...
        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- ERROR   : Test Error Level (matrixmultiplication.py:60)
        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- CRITICAL: Test Critical Level (matrixmultiplication.py:61)
        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- VERBOSE : Test Verbose Level
        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- VERBOSE : A:
        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- VERBOSE :     [1 2]
        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- VERBOSE :     [3 4]
        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- VERBOSE : B:
        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- VERBOSE :     [2 3 5]
        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- VERBOSE :     [4 5 6]
        [2020-04-03 14:17:18] -- matpy.matrixmultiplication -- WARNING : Matrix size exceeds 4 elements.

    """

    # Formats The spaces accommodate the different length of the words and
    # amount of detail wanted in the message:
    time_fmt = "[%(asctime)s.%(msecs)03d]"
    name_fmt = "- %(name)s -"
    pre_fmt = time_fmt + " " + name_fmt + " "

    debug_fmt = "%(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)"
    info_fmt = "%(levelname)-8s: %(message)s"
    warning_fmt = "%(levelname)-8s: %(message)s"
    error_fmt = "%(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)"
    critical_fmt = "%(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)"

    # Create format dictionary
    FORMATS = {
        logging.DEBUG: pre_fmt + debug_fmt,
        logging.INFO: pre_fmt + info_fmt,
        logging.WARNING: pre_fmt + warning_fmt,
        logging.ERROR: pre_fmt + error_fmt,
        logging.CRITICAL: pre_fmt + critical_fmt
    }

    # Initialize with a default logging.Formatter
    def __init__(self):
        super().__init__(fmt="- %(name)s - %(levelname)s: %(message)s",
                         datefmt=None, style='%')

    def format(self, record):

        # Use the logging.LEVEL to get the right formatting
        log_fmt = self.FORMATS.get(record.levelno)

        # Create new formatter with modified timestamp formatting.
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")

        # Return
        return formatter.format(record)
