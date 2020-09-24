# External
import os
import os.path as p
from glob import glob

# Internal
from .stationxml2STATIONS import stationxml2STATIONS


def stationxmldir2STATIONSdir(stationxmldir: str, stationsdir: str):
    """
    Takes in the path to a stationxml directory and outputs the corresponding
    station files in the stationsdir.

    Args:
        stationxmldir (str):
            Path to StationXML directory
        stationsdir (str):
            Path to STATIONS directory
    """

    # Glob the files
    stationxmlfiles = glob(p.join(stationxmldir, "*"))

    # Create path to stations directory
    for _xmlfile in stationxmlfiles:
        outname = p.join(stationsdir,
                         p.basename(_xmlfile).split(".")[0] + ".stations")
        stationxml2STATIONS(_xmlfile, outname)
