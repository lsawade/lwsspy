from typing import Tuple
from obspy import Stream
import numpy as np


def stream_grad_and_hess_win(data: Stream, synt: Stream, dsyn: Stream) \
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

    g = 0.0
    h = 0.0
    for tr in data:
        network, station, component = (
            tr.stats.network, tr.stats.station, tr.stats.component)

        # Get the trace sampling time
        dt = tr.stats.delta
        d = tr.data

        try:
            s = synt.select(network=network, station=station,
                            component=component)[0].data
            dsdz = dsyn.select(network=network, station=station,
                               component=component)[0].data

            for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                wsyn = s[win.left:win.right]
                wobs = d[win.left:win.right]
                wdsdz = dsdz[win.left:win.right]
                g += np.sum((wsyn - wobs) * wdsdz * tap) * dt
                h += np.sum(wdsdz ** 2 * tap) * dt

        except Exception as e:
            print(f"When accessing {network}.{station}.{component}")
            print(e)

    return g, h
