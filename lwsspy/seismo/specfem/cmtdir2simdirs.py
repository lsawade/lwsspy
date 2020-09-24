from glob import glob
from os import path as p
from typing import Union

from .cmt2simdir import cmt2simdir
from ..source import CMTSource


def cmtdir2simdirs(cmtdir: str, specfemdir: str, outputdir: str = "./",
                   specfem_dict: Union[dict, None] = None,
                   stationsdir: Union[str, None] = None):
    """Wrapper arount ``cmt2simdir`` to process an entire directory

    Args:
        cmtdir (str): Directory of CMT solutions
        specfemdir (str):
            Path to specfem dir
        outputdir (str, optional):
            Output Directory. Defaults to "./".
        specfem_dict (Union[dict, None], optional):
            Optional dictionary. Defaults to None which gives the following
            copy dirtree dict from the ``createsimdir`` function:

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

    # Get list of cmt files
    cmtfiles = glob(p.join(cmtdir, "*"))

    print("Number of CMT files: %d" % len(cmtfiles))

    for i, _file in enumerate(cmtfiles):

        # Create Simdir
        if stationsdir is not None:
            cmtid = p.basename(_file).split(".")[0].split("_")[0]
            stationsfile = p.join(stationsdir, cmtid + ".stations")

            if p.exists(stationsfile) is False:
                stationsfile = p.join(stationsdir, cmtid)
            if p.exists(stationsfile) is False:
                raise ValueError(f"{stationsfile} does not exist.")

        # Print progress
        print(f"#{i+1:0>5}/{len(cmtfiles)}:{_file:_>50}")

        # Create simdir
        cmt2simdir(_file, specfemdir, outputdir=outputdir,
                   specfem_dict=specfem_dict,
                   stationsfile=stationsfile)
