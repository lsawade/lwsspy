"""

Usage: 

    touchdir <path>

This sets an executable that touches all files in a directory to update them.
It resets the time stamps. Important on systems that have automatic purging 
in place for files you want to protect.


--- 

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

from subprocess import check_call
from sys import argv, exit


def bin():

    if (len(argv) != 2) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(__doc__)
        exit()
    else:
        dirpath = argv[1]

    cmddires = f"find {dirpath} -type d -exec touch '{{}}' +"
    cmdfiles = f"find {dirpath} -type f -exec touch '{{}}' +"

    # Touch all directories
    check_call(cmddires, shell=True)

    # Touch all files
    check_call(cmdfiles, shell=True)
