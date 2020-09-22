

from .readfile import readfile


def cat(filename: str):
    """Like bash's `cat` prints content of file to terminal.

    Args:
        filename (str): Filename

    Returns:
        None

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)
    """

    print(readfile(filename))
