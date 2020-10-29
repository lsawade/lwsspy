"""
STDOUT/STERR helpers
"""

import sys
import contextlib


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


def print_bar(string):
    # Running forward simulation
    print("\n\n")
    print(72 * "=")
    print(f"{f' {string} ':=^72}")
    print(72 * "=")
    print("\n")


def print_action(string):
    print(f"---> {string} ...")


def print_section(string):
    # Running forward simulation
    print("\n")
    print(f"{f' {string} ':=^72}")
    print("\n")
