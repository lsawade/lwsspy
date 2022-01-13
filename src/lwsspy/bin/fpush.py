#!/usr/bin/env python
"""

This file contains a script that if located on princeton servers copies 
to tigress/lsawade temp folder, and if not on princeton servers secure copies to
the same folder. This works only if used with VPN or tigressgateway.

"""

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
        print("You need to provide a filename. Usage: fpush <filenname/dirname>")
        exit()
    else:
        filename = argv[1]

    if "princeton" in current_host:
        check_call(
            f'cp -r {tempfolder}/{filename} {filename} ', shell=True)
    else:
        check_call(
            f'scp -r {username}@{hostname}:{tempfolder}/{filename} {filename}',
            shell=True)
