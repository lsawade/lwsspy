from numpy import arctan2, sin, cos, degrees, radians


def bearing(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dL = lon2 - lon1

    X = cos(lat2) * sin(dL)
    Y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dL)

    return degrees(arctan2(X, Y))
