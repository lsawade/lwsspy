import os
from typing import Optional
from copy import deepcopy
import _pickle as cPickle
from lwsspy import CMTSource
from lwsspy import stream_multiply
from lwsspy import read_output_traces
from lwsspy import read_measurements
from .measurements import get_all_measurements
from .io import write_fixed_traces
import numpy as np


def get_ratio(measurement_dict):

    ratiodict = dict()
    neldict = dict()
    for _wtype, _wtypedict in measurement_dict.items():
        ratiodict[_wtype] = dict()
        neldict[_wtype] = dict()
        for _comp, _compdict in _wtypedict.items():
            nel = len(_compdict['dlna'])
            ratio = np.mean(np.sqrt(np.exp(2 * np.array(_compdict['dlna']))))
            ratiodict[_wtype][_comp] = ratio
            neldict[_wtype][_comp] = nel

    return ratiodict, nel


def print_ratiodict(ratiodict: dict):

    ratios = []
    for _wtype, _wtypedict in ratiodict.items():
        for _comp in _wtypedict.keys():
            ratios.append(ratiodict[_wtype][_comp])
            print(f"{_wtype:7} {_comp} {ratiodict[_wtype][_comp]:6.4f}")
        print("")

    print(f"Full:   d{np.mean(ratios):6.4f}\n")
    print(f"Actual: {get_factor_from_ratiodict(ratiodict):6.4f}\n")


def get_factor_from_ratiodict(ratiodict, neldict):
    """Computes the weighted average of the ratios across components"""

    ratios = []
    nel = 0
    if "mantle" in ratiodict:

        for (_comp, _rat), (_, _nel) in zip(
                ratiodict["mantle"].items(), neldict["mantle"].items()):

            nel += _nel
            ratios.append(_rat * float(_nel))

        ratios = np.array(ratios)/float(nel)

    else:
        for (_wtype, _compdict), (_, _nelcompdict) in zip(
                ratiodict.items(), neldict.items()):

            for (_comp, _rat), (_, _nel) in zip(
                    _compdict.items(), _nelcompdict.items()):

                nel += nel
                ratios.append(_rat * float(_nel))

        ratios = np.array(ratios)/float(nel)

    return np.mean(ratios)


def multiply_synt(synt, factor):
    fix_synt = deepcopy(synt)
    for _, _compdict in fix_synt.items():
        for _, _stream in _compdict.items():

            stream_multiply(_stream, factor)
    return fix_synt


def fix_source(event: CMTSource, factor: float) -> CMTSource:

    event = deepcopy(event)
    M0 = event.M0
    event.M0 = factor * M0

    return event


def fix_synthetics(cmtdir, label: Optional[str] = None, verbose=True):

    # Set label
    if label is not None:
        label = "_" + label
    else:
        label = ""

    # Get output traces
    try:
        obsd, synt = read_output_traces(cmtdir)
    except Exception as e:
        if verbose:
            print(f'Couldnt read traces for {cmtdir} because {e}.')
        return -1

    # Get event
    try:
        eventfile = os.path.join(cmtdir, os.path.basename(cmtdir) + label)
        event = CMTSource.from_CMTSOLUTION_file(eventfile)
    except Exception as e:
        if verbose:
            print(f'Couldnt read event for {cmtdir} because {e}.')
        return -1

    # Measure the traces
    measurementdict_prefix = get_all_measurements(obsd, synt, event)

    # Get factor
    ratiodict, neldict = get_ratio(measurementdict_prefix)
    factor = get_factor_from_ratiodict(ratiodict, neldict)
    if verbose:
        print(f"Correction factor: {factor}")

    if np.isnan(factor) or np.isinf(factor):
        if verbose:
            print(
                f'Couldnt find good factor for {cmtdir} because factor = {factor}.'
            )
        return -1

    # Fix the traces
    fix_synt = multiply_synt(synt, factor)

    # Fix event
    fix_event = fix_source(event, factor)

    # Measure the traces after fixing
    measurementdict_fix = get_all_measurements(obsd, fix_synt, event)

    # Write the new measurement dictionary
    # Create filename
    filename = f"measurements{label}_fix.pkl"
    outfile = os.path.join(cmtdir, filename)
    if verbose:
        print(f"Outfile: {outfile}")

    # Write to measurement pickle
    with open(outfile, "wb") as f:
        cPickle.dump(measurementdict_fix, f)

    # Write the fixed synthetics to file
    write_fixed_traces(cmtdir, fix_synt)

    # Write fixed cmt solution
    eventout = os.path.join(cmtdir, os.path.basename(cmtdir) + label + "_fix")
    if verbose:
        print(f"Fixed event: {eventout}")
    fix_event.write_CMTSOLUTION_file(eventout)

    return obsd, fix_synt, measurementdict_fix


def fix_database(database: str, label: Optional[str] = None):

    cmts = os.listdir(database)

    for event in cmts:

        # Get file
        cmtdir = os.path.join(database, event)

        # Fix synthetics
        fix_synthetics(cmtdir, label=label, verbose=True)


def bin_fix_event():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='event',
                        help='event directory',
                        type=str)
    parser.add_argument('-l', '--label', dest='label',
                        type=str, default=None, required=False)

    args = parser.parse_args()

    # Fix dlna database
    fix_synthetics(args.event, label=args.label, verbose=True)


def bin_fix_dlna_database():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='database',
                        help='Database directory',
                        type=str)
    parser.add_argument('-l', '--label', dest='label',
                        type=str, default=None, required=False)

    args = parser.parse_args()

    # Fix dlna database
    fix_database(args.database, label=args.label)
