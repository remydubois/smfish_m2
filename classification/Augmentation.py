import numpy
from parameters import input_shape
import itertools
from copy import copy
import random

"""
    Data augmentation operations performed on indices only. It is much lighter than using actual image augmentation
    scripts performed on real-sized images, especially for zooming, centering, translating (shifting).
"""


def _translate(indices, steps):
    """
    Run time  5.68 µs ± 213 ns.
    :param indices:
    :param steps:
    :return:
    """
    return indices + steps


def _center(indices, shape=input_shape()):
    """
    Run time 27.8 µs ± 302 ns
    :param indices:
    :param shape:
    :return:
    """
    indices = (indices * numpy.array([[128/600],[128/600],[1]])).astype(int)
    m = indices.max()
    steps = (numpy.array([s - m for s, m in zip([*shape, 3], [m, m, 3])], dtype=numpy.int16) // 2).reshape(-1, 1)
    return _translate(indices, steps)


def _transpose(indices):
    """
    Run time 3.41 µs ± 64.6 ns
    :param indices:
    :return:
    """
    return indices[[1, 0, 2]]


def _zoom(indices, ratio):
    """
    Run time 26.6 µs ± 1.22 µs. Approx 1000x faster than a scipy's zoom or skimage rescale.
    :param indices:
    :param ratio:
    :return:
    """
    if ratio != 1.0:
        means = numpy.mean(numpy.hstack(indices), axis=1).reshape(-1, 1)
        return [means + ratio * (ch - means) for ch in indices]
    else:
        return indices


def _flip(indices, axis=0, shape=input_shape()):
    """
    Run time 4.97 µs ± 422 ns.
    :param indices:
    :param axis:
    :param shape:
    :return:
    """

    cop = indices.copy()
    cop[axis] = shape[axis] - cop[axis]
    return cop


def rot0(indices):
    return indices


def rot90(indices):
    return _transpose(_flip(indices, axis=0))


def rot180(indices):
    return _flip(_flip(indices, axis=1), axis=0)


def rot270(indices):
    return _flip(_transpose(indices), axis=0)


def augmentor(*operators):
    """
    TODO Must be compatible with three channels. As of now _center is not because the translation steps must be identical
    for each channel so as to keep image distances the same.
    :param operators:
    :return:
    """

    def f(iter):
        return map(lambda example: (operators[random.randint(0, len(operators) - 1)](_center(example[0])), ) + tuple(example[1:]),
                   iter)
        # return iter

    return f
