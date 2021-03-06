from typing import Tuple, List
from obspy import Stream
import numpy as np


def stream_grad_and_hess_win(data: Stream, synt: Stream, dsyn: List[Stream],
                             normalize: bool = True, verbose: float = False) \
        -> Tuple[float, float]:
    """Computes the gradient and the approximate hessian of the cost function
    using the Frechet derivative of the forward modelled data.
    The stats object of the Traces in the stream _*must*_ contain both
    `windows` and the `tapers` attributes!

    Parameters
    ----------
    data : Stream
        data
    synt : Stream
        synthetics
    dsyn : Stream
        frechet derivatives

    Returns
    -------
    Tuple[float, float]
        Gradient, Approximate Hessian

    Last modified: Lucas Sawade, 2020.09.28 19.00 (lsawade@princeton.edu)
    """

    g = np.zeros(len(dsyn))
    h = np.zeros(len(dsyn), len(dsyn))

    for tr in data:
        network, station, component = (
            tr.stats.network, tr.stats.station, tr.stats.component)

        # Get the trace sampling time
        dt = tr.stats.delta
        d = tr.data

        try:
            s = synt.select(network=network, station=station,
                            component=component)[0].data
            # Create trace list for the Frechet derivatives
            dsdm = []
            for ds in dsyn:
                dsdm.append(ds.select(network=network, station=station,
                                      component=component)[0].data)

            # Loop over windows
            for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                # Get data in windows
                wsyn = s[win.left:win.right]
                wobs = d[win.left:win.right]

                # Normalization factor on window
                factor = np.sum(tap * wobs ** 2) * dt

                # Compute Gradient
                for _i, _dsdm_i in enumerate(dsdm):
                    # Get derivate with respect to model parameter i
                    wdsdm_i = _dsdm_i[win.left:win.right]
                    gw = np.sum(((wsyn - wobs) * tap) * wdsdm_i) * dt
                    if normalize:
                        gw /= factor
                    g[_i] += gw

                    for _j, _dsdm_j in enumerate(dsdm):
                        # Get derivate with respect to model parameter j
                        wdsdm_j = _dsdm_j[win.left:win.right]
                        hw = ((wdsdm_i * tap) @ (wdsdm_j * tap)) * dt
                        if normalize:
                            hw /= factor
                        h[_i, _j] += hw

        except Exception as e:
            if verbose:
                print(f"When accessing {network}.{station}.{component}")
                print(e)

    return g, h
