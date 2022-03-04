import os
from subprocess import check_call
import socket
from sys import argv, exit

"""

Usage: 

    gpull-<remote> <local/path> <remote/path>

This file defines multiple scripts that sync directories from a specfic host 
with a local one using rsync. Gpull works better in the sense that it is really 
hard to push to certain platforms (SUMMIT). Pulling towards such platforms 
circumvents this issue.

Example:

    gpull-tiger <local/path> <path/on/tiger/to/dir>


:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2021.01.17 11.45

"""


def gpull(remote, remotepath, path='.'):

    # User IDs
    userid = 'lsawade'

    # Define command
    cmd = f'rsync -a --exclude=.git {userid}@{remote}:{remotepath} {path}'

    # Execute
    check_call(cmd, shell=True)


def gpull_tiger():

    # Get command line arguments
    # Check if argument is given
    if len(argv) != 3:
        print(__doc__)
        exit()
    else:
        remotepath, path = argv[1:]

    # Define fixed tiger location
    remote = 'tigergpu.princeton.edu'

    gpull(remote, remotepath, path)
