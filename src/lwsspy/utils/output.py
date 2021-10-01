"""
STDOUT/STERR helpers
"""

import sys
import contextlib
from typing import Callable


class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    """Context manager that suppresses stdout of a function.
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def print_bar(string: str):
    """Prints Bar divider for code output.

    WARNING
    -------
    Will be deprecated in favor of log_bar


    Parameters
    ----------
    string : str
        string to be output


    Notes
    -----

    ::

        ========================================================================
        ============================= <string> =================================
        ========================================================================


    """
    log_bar(string)


def log_bar(string, plogger: Callable = print):
    """Prints Bar divider for code output using logging or print. Default is 
    print.

    Parameters
    ----------
    string : str
        string to be output
    plogger : Callable
        function to use for printing


    Notes
    -----

    ::

        ========================================================================
        ============================= <string> =================================
        ========================================================================


    """
    # Running forward simulation
    plogger(" ")
    plogger(72 * "=")
    plogger(f"{f' {string} ':=^72}")
    plogger(72 * "=")
    plogger(" ")


def print_action(string: str):
    """Prints action statement.

    WARNING
    -------
    Will be deprecated in favor of log_action


    Parameters
    ----------
    string : str
        string to be output


    Notes
    -----

    ::

        ---> <string>

    """
    log_action(string)


def log_action(string: str, plogger: Callable = print):
    """Prints action statement or logs it depending on setting.

    Parameters
    ----------
    string : str
        string to be output
    plogger : Callable
        function to use for printing



    Notes
    -----

    ::

        ---> <string>

    """
    plogger(f"---> {string} ...")


def print_section(string: str):
    """Prints section divider for code output

    WARNING
    -------
    Will be deprecated in favor of log_section

    Parameters
    ----------
    string : str
        string to be output

    Notes
    -----

    Print statement format:

    ::

        ============================= <string> =================================

    """
    # Running forward simulation
    log_section(string)


def log_section(string: str, plogger: Callable = print):
    """Prints section divider for code output printed pby either a logger or 
    print function by default.

    Parameters
    ----------
    string : str
        string to be output

    Notes
    -----

    Print statement format:

    ::

        ============================= <string> =================================

    """

    # Running forward simulation
    plogger(" ")
    plogger(f"{f' {string} ':=^72}")
    plogger(" ")
