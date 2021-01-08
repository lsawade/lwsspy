import datetime as dt


def year2date(date: float) -> dt.datetime:
    """Converts a year with decimal positions to a datetime object.

    Parameters
    ----------
    date : float
        Year with decimal position

    Returns
    -------
    dt.datetime
        Output Datetime object

    Notes
    -----

    Adapted from https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.01.07 13.30


    """

    def s(date):
        # returns seconds since epoch
        return (date - dt.datetime(1900, 1, 1)).total_seconds()

    year = int(date)
    yearFraction = float(date) - int(date)
    startOfThisYear = dt.datetime(year=year, month=1, day=1)
    startOfNextYear = dt.datetime(year=year+1, month=1, day=1)
    secondsInYear = (s(startOfNextYear) -
                     s(startOfThisYear)) * yearFraction

    newdate = dt.datetime(year=year, month=1, day=1) \
        + dt.timedelta(seconds=secondsInYear)
    return newdate
