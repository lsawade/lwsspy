import os
import shutil


def cp(source: str, destination: str, ow: bool = False, verbose: bool = False,
       dry: bool = False):
    """Copies single file to new destination.
    But checking if the destination exists.

    Args:
        source (str):
            Source file
        destination (str):
            Copy destination.
        ow (bool, optional):
            Overwrite destination. Defaults to False.
        verbose (bool, optional):
            Verbose flag. Defaults to False.
        dry (bool, optional):
            Dry run flag. Defaults to False.
    
    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)
    """

    # change verbose to true if we perform a dryrun
    if dry:
        verbose = True

    # Check if path exist
    if os.path.exists(destination):
        if ow:
            if os.path.isfile(destination):
                if verbose:
                    print("%s exists and is removed." % (destination))
                if not dry:
                    os.remove(destination)
            else:
                print("Destination not a file.")
                return
        else:
            print("File exists. Not overwritten.")
            return

    if verbose:
        print("Copying %s to %s" % (source, destination))

    # Copying
    if not dry:
        shutil.copy(source, destination)
