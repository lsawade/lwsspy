

from .readfile import readfile


def cat(filename: str):
    """Like bash's `cat` prints content of file to terminal.

    Args:
        filename (str): Filename

    Returns:
        None

    """

    print(readfile(filename))