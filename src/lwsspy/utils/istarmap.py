import multiprocessing


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = multiprocessing.pool.Pool._get_tasks(
        func, iterable, chunksize)
    result = multiprocessing.pool.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          multiprocessing.pool.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


setattr(multiprocessing.pool.Pool, "istarmap", istarmap)
