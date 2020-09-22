import os
import shutil


def cpdir(source: str, destination: str, ow=False, verbose=False, dry=False):
    """Copies one directory to another. But checking if the destination exists.

    Args:
        source (str):
            Source directory
        destination (str):
            Destination directory
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

    if os.path.exists(destination):
        if ow:
            if os.path.isdir(destination):
                if verbose:
                    print("%s exists and is removed." % (destination))
                if not dry:
                    shutil.rmtree(destination)
            else:
                print("Destination not a directory.")
                return
        else:
            print("File exists. Not overwritten.")
            return
    if verbose:
        print("Recursively copying %s to %s " % (source, destination))

    # Copying
    if not dry:
        shutil.copytree(source, destination)
