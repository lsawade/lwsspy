#!/usr/bin/env python
"""

Usage:

    fpush filename

The filename can contain wildcards, but must then be enclosed in quotes.

The script contains a script that if located on princeton servers copies
to tigress/lsawade temp folder, and if not on princeton servers secure copies
to the same folder. This works only if used with VPN or tigressgateway.


:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2021.01.17 11.45

"""

import os
from subprocess import check_call
import socket
from sys import argv, exit


def bin():
    # Server Location
    username = "lsawade"
    hostname = "tigressdata.princeton.edu"
    tempfolder = "/tigress/lsawade/temp"

    # Current Host
    current_host = socket.gethostname()

    # Check if argument is given
    if len(argv) == 1:
        print(__doc__)
        exit()
    else:
        filename = argv[1:]

        if isinstance(filename, list):
            filename = " ".join(filename)

    check_call(
        f'scp -r {filename} {username}@{hostname}:{tempfolder}/',
        shell=True)


if __name__ == "__main__":
    bin()
