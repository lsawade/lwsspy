import datetime as dt
import time


def toYearFraction(date) -> float:
    """Converts date to year fraction

    Parameters
    ----------
    date : datetime.datetime
        Date

    Returns
    -------
    float
        Output year decimal

    Notes
    -----

    Adapted from: https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.07 13.30

    """

    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = dt.datetime(year=year, month=1, day=1)
    startOfNextYear = dt.datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction
