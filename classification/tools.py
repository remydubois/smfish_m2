import time
import threading
import numpy


class ThreadSafeIter(object):
    """
    Defines a thread-safe class of Iterators (therefore of generators).
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """
    A decorator for ease of use.
    :param f:
    :return:
    """

    def g(*args, **kwargs):
        return ThreadSafeIter(f(*args, **kwargs))

    return g


