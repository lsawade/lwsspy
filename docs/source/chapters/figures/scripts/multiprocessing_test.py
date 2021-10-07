import os
import multiprocessing as mp
import random
import numpy as np
from obspy import Stream, read, read_inventory, Inventory, read_events
from obspy.clients.syngine import Client

import lwsspy.seismo as lseis
import lwsspy.utils as lutils
import lwsspy.base as lbase


def get_data(cmtfilename, outputdir):

    # Download
    if os.path.exists(outputdir) is False:
        lseis.download_waveforms_cmt2storage(
            cmtfilename, outputdir, duration=7200.0)

    # Load
    st = read(os.path.join(outputdir, "waveforms/*"))
    inv = lseis.read_inventory(os.path.join(outputdir, "stations/*"))

    return st, inv


def get_modes():
    syntfiles = os.path.join(lbase.DOCFIGURESCRIPTDATA,
                             "C201711191509A.1D.sac", "[!CMTSOLUTION]*")

    synt = read(syntfiles)

    return synt


def get_synt(cmtfilename, outputdir, inventory):

    # Cmtsource
    cmtsource = lseis.CMTSource.from_CMTSOLUTION_file(cmtfilename)

    # Get synthetic seismograms
    client = Client()

    # Download Dict
    ddict = dict(
        model="ak135f_5s",
        components="RTZ",
        sourcelatitude=cmtsource.latitude,
        sourcelongitude=cmtsource.longitude,
        sourcedepthinmeters=cmtsource.depth_in_m,
        sourcemomenttensor=cmtsource.tensor * 1e-7,  # to Nm
        origintime=cmtsource.origin_time,
        starttime=cmtsource.origin_time - 30.0,
        endtime=cmtsource.origin_time + 3700.0,
        units='displacement',
        scale=1.0,
    )

    # List of stations
    bulk = [dict(
        networkcode=f"{_net}", stationcode=f"{_sta}",
        latitude=_lat, longitude=_lon)
        for _net, _sta, _lat, _lon
        in zip(*lseis.inv2net_sta(inv), *lseis.inv2geoloc(inv))]

    # Get bulk waveforms
    syntdir = os.path.join(outputdir, "synt")
    syntfilename = os.path.join(syntdir, "synt.mseed")

    # Download if synthetic is now downloaded yet
    if os.path.exists(syntfilename) is False:

        # Create directory if not existent
        if os.path.exists(syntdir) is False:
            # Make directory
            os.makedirs(syntdir)

        # Download new data
        st = client.get_waveforms_bulk(bulk=bulk, **ddict,
                                       filename=syntfilename)
    st = read(syntfilename)

    return st


if __name__ == '__main__':
    # Number of cores for processing
    nprocs = 5

    with mp.Pool(processes=nprocs) as pool:

        # Event data
        eventfile = "/Users/lucassawade/OneDrive/Python/lwsspy/C201711191509A"
        cmtsource = lseis.CMTSource.from_CMTSOLUTION_file(eventfile)
        lutils.print_action("Getting data")

        # Get the data
        with lutils.Timer():
            data, inv = get_data(eventfile, "testmulti")
            print(f"# Traces:   {len(data)}")
            print(f"# Channels: {len(inv.get_contents()['channels'])}")
            print('Reading done.')

        processdict = dict(
            inventory=inv,
            remove_response_flag=True,
            water_level=100.0,
            filter_flag=True,
            # pre_filt=[1/150.0, 1/100.0, 1/60.0, 1/50.0],
            pre_filt=[0.00285, 0.00333, 0.00667, 0.008],
            resample_flag=True,
            sampling_rate=1.0,
            taper_type='hann',
            rotate_flag=True,
            event_latitude=cmtsource.latitude,
            event_longitude=cmtsource.longitude,
            geodata=True,
            sanity_check=True,
            starttime=cmtsource.cmt_time + 5,
            endtime=cmtsource.cmt_time + 3600.0
        )

        # Process the data
        lutils.print_action("Processing data")
        pdata = lseis.multiprocess_stream(
            data, processdict, nprocs=nprocs, pool=pool)

        # Get the synthetics
        lutils.print_action("Getting synthetics")

        modes = True
        with lutils.Timer():
            if modes:
                synt = get_modes()
            else:
                synt = get_synt(eventfile, os.path.join(
                    lbase.DOCFIGURESCRIPTS, "testmulti"), inv)
            print(f"# Traces:   {len(synt)}")
            print('Reading done.')

        with lutils.Timer():
            # Get max number of samples
            npts = []
            for _tr in synt:
                npts.append(_tr.stats.npts)

            max_npts = int(np.max(npts))
            for _tr in synt:
                _tr.data.resize(max_npts)

        # Process the synthetics
        lutils.print_action("Processing synthetics")
        processdict["remove_response_flag"] = False
        processdict["rotate_flag"] = False
        psynt = lseis.multiprocess_stream(
            synt, processdict, nprocs=nprocs, pool=pool)

        window_dict = dict(
            station=inv,
            event=read_events(eventfile)[0],
            _verbose=True,
            config_dict=dict(
                config={
                    "min_period": 150.0,
                    "max_period": 300.0,
                    "stalta_waterlevel": 0.085,
                    "tshift_acceptance_level": 40.0,
                    "tshift_reference": 0.0,
                    "dlna_acceptance_level": 0.75,
                    "dlna_reference": 0.0,
                    "cc_acceptance_level": 0.85,
                    "s2n_limit": 3.0,
                    "s2n_limit_energy": 3.0,
                    "window_signal_to_noise_type": "amplitude",
                    "selection_mode": "surface_waves",
                    "min_surface_wave_velocity": 2.20,
                    "max_surface_wave_velocity": 7.10,
                    "earth_model": "ak135",
                    "max_time_before_first_arrival": 100.0,
                    "max_time_after_last_arrival": 200.0,
                    "check_global_data_quality": True,
                    "snr_integrate_base": 3.5,
                    "snr_max_base": 3.0,
                    "c_0": 0.7,
                    "c_1": 3.0,
                    "c_2": 0.0,
                    "c_3a": 1.0,
                    "c_3b": 2.0,
                    "c_4a": 3.0,
                    "c_4b": 10.0,
                    "resolution_strategy": "interval_scheduling"
                },
                components=dict(
                    Z=None,
                    R=None,
                    T=None
                ),

                instrument=dict(instrument_merge_flag=True)
            )
        )

        lutils.print_action("Windowing")
        wobsd = lseis.multiwindow_stream(
            pdata.copy(), psynt, window_dict, nprocs=nprocs)

        lutils.print_action("Plotting")
        lseis.stream_pdf(
            wobsd.select(component='Z'),
            synt=psynt.select(component='Z'),
            cmtsource=cmtsource)
