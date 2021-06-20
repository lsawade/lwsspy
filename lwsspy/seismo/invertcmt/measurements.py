def get_toffset(
        tsample: int, dt: float, t0: UTCDateTime, origin: UTCDateTime) -> float:
    """Computes the time of a sample with respect to origin time

    Parameters
    ----------
    tsample : int
        sample on trace
    dt : float
        sample spacing
    t0 : UTCDateTime
        time of the first sample
    origin : UTCDateTime
        origin time

    Returns
    -------
    float
        Time relative to origin time
    """

    # Second on trace
    trsec = (tsample*dt)
    return (t0 + trsec) - origin

    # Normalize by component and aximuthal weights


def get_measurements_and_windows(
        obs: Stream, syn: Stream, event: CMTSource):

    windows = dict()

    # Create dict to access traces
    for _component in ["R", "T", "Z"]:
        windows[_component] = dict()
        windows[_component]["id"] = []
        windows[_component]["dt"] = []
        windows[_component]["starttime"] = []
        windows[_component]["endtime"] = []
        windows[_component]["nsamples"] = []
        windows[_component]["latitude"] = []
        windows[_component]["longitude"] = []
        windows[_component]["distance"] = []
        windows[_component]["azimuth"] = []
        windows[_component]["back_azimuth"] = []
        windows[_component]["nshift"] = []
        windows[_component]["time_shift"] = []
        windows[_component]["maxcc"] = []
        windows[_component]["dlna"] = []
        windows[_component]["L1"] = []
        windows[_component]["L2"] = []
        windows[_component]["dL1"] = []
        windows[_component]["dL2"] = []
        windows[_component]["trace_energy"] = []
        windows[_component]["L1_Power"] = []
        windows[_component]["L2_Power"] = []

        for _tr in obs:
            if _tr.stats.component == _component \
                    and "windows" in _tr.stats:

                d = _tr.data
                try:
                    network, station, component = (
                        _tr.stats.network, _tr.stats.station,
                        _tr.stats.component)
                    s = syn.select(
                        network=network, station=station,
                        component=component)[0].data
                except Exception as e:
                    self.logger.warning(
                        f"{network}.{station}..{component}")
                    self.logger.error(e)
                    continue

                trace_energy = 0
                for win in _tr.stats.windows:
                    # Get window data
                    wd = d[win.left:win.right]
                    ws = s[win.left:win.right]

                    # Infos
                    dt = _tr.stats.delta
                    npts = _tr.stats.npts
                    winleft = get_toffset(
                        win.left, dt, win.time_of_first_sample,
                        event.origin_time)
                    winright = get_toffset(
                        win.right, dt, win.time_of_first_sample,
                        event.origin_time)

                    # Populate the dictionary
                    windows[_component]["id"].append(_tr.id)
                    windows[_component]["dt"].append(dt)
                    windows[_component]["starttime"].append(winleft)
                    windows[_component]["endtime"].append(winright)
                    windows[_component]["latitude"].append(
                        _tr.stats.latitude
                    )
                    windows[_component]["longitude"].append(
                        _tr.stats.longitude
                    )
                    windows[_component]["distance"].append(
                        _tr.stats.distance
                    )
                    windows[_component]["azimuth"].append(
                        _tr.stats.azimuth
                    )
                    windows[_component]["back_azimuth"].append(
                        _tr.stats.back_azimuth
                    )

                    # Measurements
                    max_cc_value, nshift = lpy.xcorr(wd, ws)

                    # Get fixed window indeces.
                    istart, iend = win.left, win.right
                    istart_d, iend_d, istart_s, iend_s = lpy.correct_window_index(
                        istart, iend, nshift, npts)
                    wd_fix = d[istart_d:iend_d]
                    ws_fix = s[istart_s:iend_s]

                    powerl1 = lpy.power_l1(wd, ws)
                    powerl2 = lpy.power_l2(wd, ws)
                    norm1 = lpy.norm1(wd)
                    norm2 = lpy.norm2(wd)
                    dnorm1 = lpy.dnorm1(wd, ws)
                    dnorm2 = lpy.dnorm2(wd, ws)
                    dlna = lpy.dlna(wd_fix, ws_fix)
                    trace_energy += norm2

                    windows[_component]["L1"].append(norm1)
                    windows[_component]["L2"].append(norm2)
                    windows[_component]["dL1"].append(dnorm1)
                    windows[_component]["dL2"].append(dnorm2)
                    windows[_component]["dlna"].append(dlna)
                    windows[_component]["L1_Power"].append(powerl1)
                    windows[_component]["L2_Power"].append(powerl2)
                    windows[_component]["nshift"].append(nshift)
                    windows[_component]["time_shift"].append(
                        nshift * dt
                    )
                    windows[_component]["maxcc"].append(
                        max_cc_value
                    )
                # Create array with the energy
                windows[_component]["trace_energy"].extend(
                    [trace_energy]*len(_tr.stats.windows))

    return windows
