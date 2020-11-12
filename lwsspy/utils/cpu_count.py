import os


def cpu_count():
    """Returns the number of CPUs in the system"""
    num = os.cpu_count()
    if num is None:
        raise NotImplementedError('Cannot determine number of cpus')
    else:
        return num
