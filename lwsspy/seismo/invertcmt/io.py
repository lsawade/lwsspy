
"""
Just a few scripts to read outputs of the the inversion

"""

import os
import glob
import _pickle as pickle
from copy import deepcopy
import _pickle as cPickle


def read_traces(wtype, streamdir):
    with open(os.path.join(streamdir, f"{wtype}_stream.pkl"), 'rb') as f:
        d = pickle.load(f)
    return d


def read_output_traces(cmtdir: str, verbose: bool = True):
    """Given an Inversion directory, read the output waveforms

    Parameters
    ----------
    cmtdir : str
        Inversion directory
    verbose : str
        Print errors/warnings

    Returns
    -------
    Tuple(dict,dict)
        Contains all wtypes available and the respective components.

    """

    # Get the output directory
    outputdir = os.path.join(cmtdir, "output")
    observeddir = os.path.join(outputdir, "observed")
    syntheticdir = os.path.join(outputdir, "synthetic")

    # Glob all wavetype
    wavedictfiles = glob.glob(os.path.join(observeddir, "*_stream.pkl"))
    wtypes = [os.path.basename(x).split("_")[0] for x in wavedictfiles]

    # Read dictionary
    obsd = dict()
    synt = dict()

    for _wtype in wtypes:

        try:
            tobsd = read_traces(_wtype, observeddir)
            tsynt = read_traces(_wtype, syntheticdir)

            obsd[_wtype] = deepcopy(tobsd)
            synt[_wtype] = deepcopy(tsynt)

        except Exception as e:
            if verbose:
                print(f"Couldnt read {_wtype} in {cmtdir} because ")
                print(e)

    return obsd, synt

def read_measurements(cmtdir: str):

    measurement_pickle_before = os.path.join(
        cmtdir, "measurements_before.pkl")
    measurement_pickle_after = os.path.join(
        cmtdir, "measurements_after.pkl")
    try:
        with open(measurement_pickle_before, "rb") as f:
            measurements_before = cPickle.load(f)
        with open(measurement_pickle_after, "rb") as f:
            measurements_after = cPickle.load(f)

        return measurements_before, measurements_after

    except Exception:
        return None
