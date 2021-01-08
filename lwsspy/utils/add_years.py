import datetime as dt


def add_years(date: dt.datetime, years: int) -> dt.datetime:
    """Return a date that's `years` years after the date (or datetime)
    object `d`. Return the same calendar date (month and day) in the
    destination year, if it exists, otherwise use the following day
    (thus changing February 29 to March 1).

    Notes
    -----
    Adapted from: https://stackoverflow.com/questions/15741618/add-one-year-in-current-date-python/15743908

    Parameters
    ----------
    date : dt.datetime
        date to add a year to 
    years : int
        number of years to add

    Returns
    -------
    datetime.datetime
       date with added year in form of datetime object 
    """

    try:
        return date.replace(year=date.year + years)
    except ValueError:
        return date + (dt.datetime(date.year + years, 1, 1)
                       - dt.datetime(date.year, 1, 1))
