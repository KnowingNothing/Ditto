from multiprocessing import Pool
import os
import time
from .logger import ditto_logger

_FUNC = None


def _WORKER_INIT(func):
    global _FUNC
    _FUNC = func


def _WORKER(x):
    return _FUNC(x)


class BaseItem(object):
    """The base item for search space."""

    pass


class BaseSpace(object):
    """The base class for space object."""

    def __init__(self):
        self.name = ""
        self.choices = None


def parallel_map(func, ids, parallel=os.cpu_count(), report_time=False):
    beg = time.time()
    with Pool(parallel, initializer=_WORKER_INIT, initargs=(func,)) as p:
        ret = p.map(_WORKER, ids)
    end = time.time()
    if report_time:
        ditto_logger.info(f"Parallel processing uses {end - beg}s.")
    return ret
