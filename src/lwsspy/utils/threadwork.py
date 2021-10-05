from scipy import spatial
from multiprocessing.pool import Pool
import numpy as np
import threading
from functools import partial
from .. import utils as lutils
import time


def function(x, y, k=0):
    """
    function to increment global variable x
    """
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
    return x,  y + k


def threadfunc(*args, function, results, it, **kwargs):
    """Thread function wrapper.

    Parameters
    ----------
    function : [type]
        [description]
    results : [type]
        [description]
    """

    results[it] = function(*args, **kwargs)


def threadwork(function, args: list or tuple, kwargs: dict or None = None,
               nargout: int = 1):
    """Embarassingly Parallel multithreading"""

    # Get CpuCount
    Ncpu = lutils.cpu_count()

    # Determine the amount of work to be done
    Nargs = len(args)
    Ntask = len(args[0])

    # Number of Chunks
    Nchunks = int(np.ceil(Ntask/Ncpu))
    chunksize = int(np.ceil(Ntask/Nchunks))

    # Cutting the arguments in chunks
    argchunks = []
    for arg in args:
        argchunks.append(lutils.chunks(arg.tolist(), chunksize))

    # Determine Kwargs,
    if type(kwargs) is None:
        kwargs = {}

    threads = []
    results = [None] * Nchunks

    for _i, args in enumerate(zip(*argchunks)):
        func = partial(threadfunc, function=function, results=results, it=_i)
        threads.append(threading.Thread(
            target=func, args=(*args,), kwargs=kwargs))

    # start threads
    for t in threads:
        t.start()

    for _i, t in enumerate(threads):
        t.join()
    #  Bringing together the results
    complete = []
    for _i in range(nargout):
        outlist = []
        for _res in results:
            outlist.extend(_res[_i])
        complete.append(outlist)

    return tuple(complete)


if __name__ == "__main__":
    import numpy as np

    x = np.linspace(0.0, 40.0, 41)
    y = np.linspace(0.0, 20.0, 41)
    results = threadwork(function, args=(x, y), kwargs={}, nargout=2)

    print(results)
    print(len(results))
