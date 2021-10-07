def fix_map_extent(extent, fraction=0.05):

    # Get extent values and fix them
    minlon, maxlon, minlat, maxlat = extent

    latb = (maxlat - minlat) * fraction
    lonb = (maxlon - minlon) * fraction

    # Max lat
    if maxlat + latb > 90.0:
        maxlat = 90.0
    else:
        maxlat = maxlat + latb

    # Min lat
    if minlat - latb < -90.0:
        minlat = -90.0
    else:
        minlat = minlat - latb

    # Max lon
    if maxlon + lonb > 180.0:
        maxlon = 180.0
    else:
        maxlon = maxlon + lonb

    # Minlon
    if minlon - lonb < -180.0:
        minlon = -180.0
    else:
        minlon = minlon - lonb

    return [minlon, maxlon, minlat, maxlat]
