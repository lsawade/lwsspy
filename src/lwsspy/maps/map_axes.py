from cartopy.crs import PlateCarree, Mollweide, UTM
import matplotlib.pyplot as plt


def map_axes(
        proj: str = "moll", central_longitude=0.0, zone: int = None,
        southern_hemisphere: bool = False) -> plt.Axes:
    """Creates matplotlib axes with map projection taken from cartopy.

    Parameters
    ----------
    proj: str, optional
        shortname for mapprojection
        'moll', 'carr', 'utm', by default "moll"
    central_longitude: float, optional
        What the name suggests default 0.0
    zone: int, optional
        if proj is 'utm', this value must be specified and refers to the UTM
        projection zone
    southern_hemisphere: bool, optional
        if proj is UTM please specify whether you want the southern or Northern
        Hemisphere by setting this flag. Default is False, which sets the option
        to Northern Hemisphere.


    Returns
    -------
    plt.Axes
        Matplotlib axes with projection

    Raises
    ------
    ValueError
        If non supported shortname for axes is given

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.01.13 20.30

    Examples
    --------

    >>> from lwsspy.plot import map_axes
    >>> map_axes()

    """

    # Check whether name is supported.
    if proj not in ['moll', 'carr', 'utm']:
        raise ValueError(f"Either 'moll' for mollweide, "
                         f"'carr' for PlateCarree or 'utm' for UTM.\n'{proj}'"
                         f"is not supported.")

    if proj == 'moll':
        projection = Mollweide(central_longitude=central_longitude)
    elif proj == 'carr':
        projection = PlateCarree(central_longitude=central_longitude)
    elif proj == 'utm':
        projection = UTM(zone, southern_hemisphere=southern_hemisphere)

    ax = plt.axes(projection=projection)

    return ax
