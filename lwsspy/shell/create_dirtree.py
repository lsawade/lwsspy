import os
from .ln import ln
from .cp import cp
from .cpdir import cpdir


def create_dirtree(root: str, dictionary: dict, dry: bool = False):
    """
    Creates paths from dictionaries structures.

    Args:
        root (str):
            location to create the rest of the structure
        dictionary (dict):
            dictionary defining the structure of the directory
        dry (bool):
            If True, no structure is created but structure is printed to
            the terminal.

    Example Dictionary:

         # Output structure
        dirs = {
                "output": {
                    "cmt3d": {
                        "waveform_plots": None,
                        "new_synt": None
                    },
                    "g3d":{
                        "waveform_plots": None,
                        "new_synt": None
                    }
                }
            }

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)

    """

    if os.path.exists(root) is False:
        os.makedirs(root)

    for key, value in dictionary.items():

        # Get path from key and root
        path = os.path.join(root, key)

        #  Check if path exists
        if os.path.exists(path):
            pass
        else:
            if dry:
                print("Test created path: %s" % path)
            else:
                os.makedirs(path)
                print("Created path: %s" % path)

        if value is not None:
            create_dirtree(path, value, dry=dry)
