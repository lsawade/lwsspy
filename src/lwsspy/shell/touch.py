

def touch(filename: str):
    """Implements bash's touch function.

    Args:
        filename (str): Filename

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)
    """
    open(filename, 'w').close()
