import dload


def downloadfile(url: str, floc: str):
    """Downloads file to location

    Parameters
    ----------
    url : str
        Source URL
    floc : str
        Destination
    """
    dload.save(url, floc)
