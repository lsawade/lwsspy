"""

Some important and simple IO help functions.

:copyright:
    Wenjie Lei (lei@princeton.edu) Year? pyasdf
    Lucas Sawade (lsawade@princeton.edu) 2019

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

"""

# External imports
from __future__ import (absolute_import, division, print_function)
import os
import json
import yaml
import logging
from pyasdf import ASDFDataSet
from obspy import Stream, Inventory
from obspy import read_inventory

# Internal imports
from ..source import CMTSource
from ..log_util import modify_logger

logger = logging.getLogger(__name__)
modify_logger(logger)


def load_json(filename):
    with open(filename) as fh:
        return json.load(fh)


def dump_json(content, filename):
    with open(filename, 'w') as fh:
        json.dump(content, fh, indent=2, sort_keys=True)


def check_dict_keys(dict_to_check, keys):
    if not isinstance(dict_to_check, dict):
        raise TypeError("input dict_to_check should be type of dict: %s"
                        % (type(dict_to_check)))

    set_input = set(dict_to_check.keys())
    set_stand = set(keys)

    if set_input != set_stand:
        print("More: %s" % (set_input - set_stand))
        print("Missing: %s" % (set_stand - set_input))
        raise ValueError("Keys is not consistent: %s --- %s"
                         % (set_input, set_stand))


def write_yaml_file(d, filename, **kwargs):
    """Writes dictionary to given yaml file.

    Args:
          d: Dictionary to be written into the yaml file
          filename: string with filename of the file to be written.

    """
    with open(filename, 'w+') as yaml_file:
        yaml.dump(d, yaml_file, default_flow_style=False, **kwargs)


def read_yaml_file(filename):
    with open(filename, "rb") as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def smart_read_yaml(yaml_file, mpi_mode=True, comm=None):
    """
    Read yaml file into python dict, in mpi_mode or not
    """
    if not mpi_mode:
        yaml_dict = read_yaml_file(yaml_file)
    else:
        if comm is None:
            comm = _get_mpi_comm()
        rank = comm.rank
        if rank == 0:
            try:
                yaml_dict = read_yaml_file(yaml_file)
            except Exception as err:
                print("Error in read %s as yaml file: %s" % (yaml_file, err))
                comm.Abort()
        else:
            yaml_dict = None
        yaml_dict = comm.bcast(yaml_dict, root=0)
    return yaml_dict


def is_mpi_env():
    """
    Test if current environment is MPI or not
    """
    try:
        import mpi4py
    except ImportError:
        return False

    try:
        import mpi4py.MPI
    except ImportError:
        return False

    if mpi4py.MPI.COMM_WORLD.size == 1 and mpi4py.MPI.COMM_WORLD.rank == 0:
        return False

    return True


def _get_mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def get_location_in_database(cmtfile, databasedir):
    """ Takes in CMT solution and database directory and outputs path to the CMT
    in the

    :param cmtfile: cmtfilename
    :param databasedir: database directory
    :return:
    """

    # Load CMT solution
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)

    # Get ID from source
    cmtID = cmtsource.eventname

    return os.path.join(os.path.abspath(databasedir),
                        "C" + cmtID,
                        "C" + cmtID + ".cmt")


def get_cmt_id(cmtfile):
    """ Takes in CMTSOLUTION file and outputs the id

    :param cmtfile: cmtfilename
    :return: ids
    """

    # Load CMT solution
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)

    # Get ID from source
    return cmtsource.eventname


def load_asdf(filename: str, no_event=False):
    """Takes in a filename of an asdf file and outputs event, inventory,
    and stream with the traces. Note that this is only good for asdffiles
    with one set of traces event and stations since the function will get the
    first/only waveform tag from the dataset

    Args:
        filename: ASDF filename. "somethingsomething.h5"

    Returns:
        Event, Inventory, Stream
    """

    ds = ASDFDataSet(filename)

    # Create empty streams and inventories
    inv = Inventory()
    st = Stream()

    # Get waveform tag
    tag = list(ds.waveform_tags)[0]
    for station in ds.waveforms.list():
        try:
            st += getattr(ds.waveforms[station], tag)
            inv += ds.waveforms[station].StationXML
        except Exception as e:
            logger.verbose(e)

    # Choose not to load an event from the asdf file (pycmt3d's event doesn't
    # output an event...)
    if not no_event:
        ev = ds.events[0]
        del ds

        return ev, inv, st
    else:
        del ds
        return inv, st


def flex_read_stations(filenames: str or list):
    """ Takes in a list of strings and tries to read them as inventories
    Creates a single inventory, not an aggregate of inventories

    :param filename: station file(s). wildcards permitted.
    :return: `obspy.Inventory`
    """

    if type(filenames) is str:
        filenames = [filenames]

    inv = Inventory()
    for _file in filenames:
        try:
            add_inv = read_inventory(_file)
            for network in add_inv:
                if len(inv.select(network=network.code)) == 0:
                    inv.networks.append(network)
                else:
                    new_network = inv.select(network=network.code)[0]
                    # print(new_network)
                    for station in network:
                        if len(new_network.select(station=station.code)) == 0:
                            new_network.stations.append(station)

                    inv = inv.remove(network=network.code)
                    inv.networks.append(new_network)

        except Exception as e:
            logger.warning("%s could not be read. Error: %s" % (_file, e))

    return inv
