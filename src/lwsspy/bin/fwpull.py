#!/usr/bin/env python
"""

This file contains a script downloads a file from a in-script-specfied server.
For now this server is the Princeton tigress-web server. On the macOS only 
curl is available. 

"""

from subprocess import check_call
import socket
from sys import argv, exit


def bin():
    # Server Location
    url = "https://tigress-web.princeton.edu/~lsawade/temp"

    # Current Host
    current_host = socket.gethostname()

    # Check if argument is given
    if len(argv) == 1:
        print("You need to provide a filename. Usage: fwpush <filenname/dirname>")
        exit()
    else:
        filename = argv[1]

    # Add more hosts that do not have wget
    if current_host in ["geo-lsawade19"]:
        check_call(
            f'curl -LO {url}/{filename}', shell=True)
    else:
        check_call(
            f' wget --no-parent -r {url}/{filename}',
            shell=True)
