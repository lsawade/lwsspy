import os
from .ln import ln
from .cp import cp
from .cpdir import cpdir
from .touch import touch


def copy_dirtree(source: str, destination: str, dictionary: dict,
                 ow: bool = True, dry: bool = False, verbose: bool = False):
    """
    Creates paths from dictionaries structures.

    Args:
        source (str):
            Source root directory
        destination (str):
            Destination root
        dictionary (dict):
            dictionary defining the structure of the directory
        ow (bool):
            Choice of overwriting. Default is ``True``.
        dry (bool):
            If ``True``, no structure is created but structure is
            printed to the terminal.

    Example Dictionary:

    The dictionary can contain different types depending
    on the file to be copied: "link", "file", "dir", None.
    "link" links old location to new location.
    "file" copies old file to new dirtree
    "dir" copies entire directory.
    None simply creates a new directory
         # Output structure
        dirs = {
            "testdir": None,
            "bin": "link"
            "DATA": {
                "Par_file": "file"
                "CMTSOLUTION": "file",
                "STATIONS": "file"
                },
            "DATABASES_MPI": "link",
            "OUTPUT_FILES": "dir"
        }

    This means that:

        `source/bin` is linked to destination `destination/bin`
        `source/DATA/Par_file` is copied to `destination/DATA/Par_file`
        ...
        `source/OUTPUT_FILES/` is copied entirely to `destination/OUTPUT_FILES/`

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)

    """
    if not os.path.exists(destination):
        if verbose:
            print("Created path: %s" % destination)
        if not dry:
            os.makedirs(destination)

    for key, value in dictionary.items():

        # Get path from key and root
        source_path = os.path.join(source, key)
        destination_path = os.path.join(destination, key)

        # Check dir for file typ
        if value == "link":
            print(f"OW: {ow}")
            ln(source_path, destination_path, ow=ow, verbose=verbose, dry=dry)
        elif value == "file":
            cp(source_path, destination_path, ow=ow, verbose=verbose, dry=dry)
        elif value == "dir":
            cpdir(source_path, destination_path, ow=ow, verbose=verbose,
                  dry=dry)
        elif value is None:
            if verbose:
                print("Created path: %s" % destination_path)
            if not dry:
                os.makedirs(source_path)
        else:
            copy_dirtree(source_path, destination_path, value, dry=dry)
