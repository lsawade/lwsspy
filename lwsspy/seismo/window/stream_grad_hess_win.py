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
    h = np.zeros(len(dsyn))

    for tr in data:
        network, station, component = (
            tr.stats.network, tr.stats.station, tr.stats.component)

        # Get the trace sampling time
        dt = tr.stats.delta
        d = tr.data

        try:
            s = synt.select(network=network, station=station,
                            component=component)[0].data

            for _i, ds in enumerate(dsyn):
                dsdm = ds.select(network=network, station=station,
                                 component=component)[0].data
                for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                    wsyn = s[win.left:win.right]
                    wobs = d[win.left:win.right]
                    wdsdm = dsdm[win.left:win.right]
                    gw = np.sum((wsyn - wobs) * wdsdm * tap) * dt
                    hw = np.sum(wdsdm * tap) * dt
                    if normalize:
                        factor = np.sum(tap * (wobs) ** 2) * dt
                        gw /= factor
                        hw /= factor
                    g[_i] += gw
                    h[_i] += hw

        except Exception as e:
            print(f"When accessing {network}.{station}.{component}")
            print(e)

    return g, h
