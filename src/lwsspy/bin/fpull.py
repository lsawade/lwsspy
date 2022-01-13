#!/usr/bin/env python
"""

This file contains a script that if located on princeton servers copies from
tigress/temp folder, and if not on Princeton servers secure copies from the same
folder. This only works if used with VPN or tigressgateway. Otherwise, Duo 
access is required.

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
        print("You need to provide a filename. Usage: fpull <filenname/dirname>")
        exit()
    else:
        filename = argv[1]

    if "princeton" in current_host:
        check_call(
            f'cp -r {filename} {tempfolder}/{filename}', shell=True)
    else:
        check_call(
            f'scp -r {filename} {username}@{hostname}:{tempfolder}/{filename}',
            shell=True)
