import os
from multiprocessing import Pool
import random
from obspy import Stream, read, read_inventory, Inventory, read_events
from obspy.clients.syngine import Client

import lwsspy as lpy


def get_data(cmtfilename, outputdir):

    # Download
    if os.path.exists(outputdir) is False:
        lpy.download_waveforms_cmt2storage(
            cmtfilename, outputdir, duration=3600.0)

    # Load
    st = read(os.path.join(outputdir, "waveforms/*"))
    inv = lpy.read_inventory(os.path.join(outputdir, "stations/*"))

    return st, inv


def get_synt(cmtfilename, outputdir, inventory):

    # Cmtsource
    cmtsource = lpy.CMTSource.from_CMTSOLUTION_file(cmtfilename)

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
        in zip(*lpy.inv2net_sta(inv), *lpy.inv2geoloc(inv))]

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

    pool = Pool(processes=nprocs)

    # Event data
    eventfile = "C201711191509A"
    cmtsource = lpy.CMTSource.from_CMTSOLUTION_file(eventfile)
    lpy.print_action("Getting data")

    # Get the data
    with lpy.Timer():
        data, inv = get_data(eventfile, "testmulti")
        print(f"# Traces:   {len(data)}")
        print(f"# Channels: {len(inv.get_contents()['channels'])}")
        print('Reading done.')

    processdict = dict(
        inventory=inv,
        remove_response_flag=True,
        water_level=100.0,
        filter_flag=True,
        pre_filt=[1/150.0, 1/100.0, 1/60.0, 1/50.0],
        resample_flag=True,
        sampling_rate=1.0,
        taper_type='hann',
        rotate_flag=True,
        event_latitude=cmtsource.latitude,
        event_longitude=cmtsource.longitude,
        geodata=False,
        sanity_check=True,
        starttime=cmtsource.origin_time,
        endtime=cmtsource.origin_time + 3570.0
    )

    # Process the data
    lpy.print_action("Processing data")
    pdata = lpy.multiprocess_stream(
        data, processdict, nprocs=nprocs, pool=pool)

    # Get the synthetics
    lpy.print_action("Getting synthetics")

    with lpy.Timer():
        synt = get_synt(eventfile, "testmulti", inv)
        print(f"# Traces:   {len(synt)}")
        print('Reading done.')

    # Process the synthetics
    lpy.print_action("Processing synthetics")
    processdict["remove_response_flag"] = False
    processdict["rotate_flag"] = False
    psynt = lpy.multiprocess_stream(
        synt, processdict, nprocs=nprocs, pool=pool)

    window_dict = dict(
        station=inv,
        event=read_events(eventfile)[0],
        config_dict=dict(
            config={
                "min_period":  40.0,
                "max_period": 100.0,
                "stalta_waterlevel": 0.085,
                "tshift_acceptance_level": 18.0,
                "tshift_reference": 0.0,
                "dlna_acceptance_level": 0.8,
                "dlna_reference": 0.0,
                "cc_acceptance_level": 0.85,
                "s2n_limit": 3.0,
                "s2n_limit_energy": 3.0,
                "window_signal_to_noise_type": "amplitude",
                "selection_mode": "body_waves",
                "min_surface_wave_velocity": 3.20,
                "max_surface_wave_velocity": 4.10,
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
                R={
                    "s2n_limit": 3.5,
                    "s2n_limit_energy": 3.5,
                    "snr_max_base": 3.5,
                },
                T={
                    "s2n_limit": 3.5,
                    "s2n_limit_energy": 3.5,
                    "snr_max_base": 3.5}
            ),

            instrument=dict(instrument_merge_flag=True)
        )
    )

    lpy.print_action("Windowing")
    wobsd = lpy.multiwindow_stream(pdata, psynt, window_dict, nprocs=nprocs)

    lpy.print_action("Plotting")
    lpy.stream_pdf(
        wobsd.select(component='Z'),
        synt=psynt.select(component='Z'),
        cmtsource=cmtsource)

    pool.close()
