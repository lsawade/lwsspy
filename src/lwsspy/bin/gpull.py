"""

Usage: 

    gpull <remote> <local/path> <remote/path>

This file defines multiple scripts that sync directories from a specfic host 
with a local one using rsync. Gpull works better in the sense that it is really 
hard to push to certain platforms (SUMMIT). Pulling towards such platforms 
circumvents this issue.

Example:

    gpull tiger testdir .

Uses rsync to pull changes from 
    <puid>@tigergpu.princeton.edu":testdir 
to 
    ./testdir


Note:

    I have hardcoded my username. If you have multiple names across devices you
    could create another command line argument that needs to be provided.

--- 

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

from subprocess import check_call
from sys import argv, exit


def sub_gpull(remote, remotepath, path='.'):

    # User IDs
    userid = 'lsawade'

    # Define command
    cmd = f'rsync -a --exclude=.git --exclude="*.egg-info" {userid}@{remote}:{remotepath} {path}'
    print(cmd)

    # Execute
    check_call(cmd, shell=True)


def gpull():

    # Get command line arguments
    # Check if argument is given
    if (len(argv) != 4) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(__doc__)
        exit()
    else:
        remote, remotepath, path = argv[1:]

    # Define fixed tiger location
    if remote == 'tiger':
        remote = 'tigergpu.princeton.edu'
    elif remote == 'traverse':
        remote = 'traverse.princeton.edu'
    # Assume it's a custom remote location
    else:
        pass

    sub_gpull(remote, remotepath, path)
