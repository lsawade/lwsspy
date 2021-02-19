from contextlib import contextmanager
import multiprocessing
from typing import List, Union


def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def multiwrapper(function, varargs,
                 varkwargs: Union[List[dict], None] = None,
                 otherkwargs: dict = dict(), sumfunc=None, processes=6,
                 d1=True):
    global func

    # This one handles variables dictionary entries
    if varkwargs is not None:
        # Handles the whether varargs has multiple dimensions
        # Has to be given manually.
        if d1:
            def func(x, y): return function(x, *otherargs, **y, **otherkwargs)
        else:
            def func(x, y): return function(*x, *otherargs, **y, **otherkwargs)
    else:
        if d1:
            def func(x): return function(x, *otherargs, **otherkwargs)
        else:
            def func(x): return function(*x, *otherargs, **otherkwargs)

    with poolcontext(processes=processes) as pool:
        if varkwargs is not None:
            results = pool.map(func, zip(varargs, varkwargs))
        else:
            results = pool.map(func, varargs)
    # In the case of a Stream, the results are Traces
    # But the output should be a trace again. So, we have to supply
    # a function that brings that back to stream form
    # def sumfunc(results): return Stream(results)
    if sumfunc is not None:
        results = sumfunc(results)
    return results
