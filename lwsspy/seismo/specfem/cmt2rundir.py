from os import path as p
from typing import Union
from ...shell.cp import cp
from .createsimdir import createsimdir


def cmt2rundir(cmtfilename: str, specfemdir: str, rundir: str,
               specfem_dict: Union[dict, None] = None):
    """Takes in ``CMTSOLUTION`` file and specfem directory and creates specfem
    simulation directory.
    Uses the ``cmtfilename`` to create new simulation directory in the
    output directory.

    Args:
        cmtfilename (str):
            Path to CMTSOLUTION
        specfemdir (str):
            Path to specfem dir
        outputdir (str, optional):
            Output Directory. Defaults to "./".
        specfem_dict (Union[dict, None], optional):
            Optional dictionary. Defaults to None which gives the following
            copy dirtree dict from the ``createsimdir`` function

            .. code:: python

                {
                    "bin": "link",
                    "DATA": {
                        "CMTSOLUTION": "file",
                        "Par_file": "file",
                        "STATIONS": "file"
                    },
                    "DATABASES_MPI": "link",
                    "OUTPUT_FILES": "dir"
                }

    Returns:
        None

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)

    """

    # Copy directory to new destination
    createsimdir(specfemdir, rundir, specfem_dict=specfem_dict)

    # Overwrite cmt solution
    cp(cmtfilename, p.join(rundir, "DATA", "CMTSOLUTION"), ow=True)
