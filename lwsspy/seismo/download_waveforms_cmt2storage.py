# External
from typing import Union
from obspy import read_inventory

# Internal
from .download_waveforms_to_storage import download_waveforms_to_storage
from .source import CMTSource
from .inv2net_sta import inv2net_sta
from .read_inventory import flex_read_inventory


def download_waveforms_cmt2storage(
        cmtfilename: str,
        *args,
        duration: float = 11000.0,
        stationxml: Union[str, None] = None,
        starttimeoffset: float = - 300.0,
        endtimeoffset: float = 0.0,
        **kwargs):

    # get startime and endtime from cmtsolution
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfilename)
    starttime = cmtsource.origin_time + starttimeoffset
    endtime = cmtsource.origin_time + endtimeoffset + duration

    if stationxml is not None:
        inv = flex_read_inventory(stationxml)
    else:
        inv = None

    download_waveforms_to_storage(*args, starttime=starttime, endtime=endtime,
                                  limit_stations_to_inventory=inv,
                                  **kwargs)
