import numpy
import math
from keras.utils import to_categorical
from skimage.transform import resize
from skimage import io
import itertools
import random
import warnings
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha=45, sigma=4.5, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = numpy.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


def distord(m):
    rs = random.randint(0, 10000)
    return [elastic_transform(n, random_state=numpy.random.RandomState(seed=rs)) for n in m]

def rot0(m):
    return m


def rot90(m):
    return [numpy.flip(n, axis=0).T for n in m]


def rot180(m):
    return [numpy.flip(numpy.flip(n, axis=1), axis=0) for n in m]


def rot270(m):
    return [numpy.flip(n.T, axis=0) for n in m]


def mirror(m):
    factor = [random.randint(-100, 100), random.randint(-100, 100)]
    shape = m[0].shape[0]
    s = numpy.arange(shape)
    inds = numpy.concatenate([s[-1:0:-1], s, s[-2:0:-1]])
    slx = inds[shape + factor[0]:2 * shape + factor[0]]
    sly = inds[shape + factor[1]:2 * shape + factor[1]]
    return [n[slx, :][:, sly] for n in m]


# def roll(m, steps=(0, 0)):
#     return [numpy.roll(numpy.roll(n, steps[0], axis=0), steps[1], axis=1) for n in m]

# def distort(m):
#     A = m.shape[0] / 80.0
#     w = 4.0 / m.shape[1]
#     shift = A * numpy.sin(2.0 * numpy.pi * numpy.arange(512) * w)
#     for i in range(512):
#         m[:, i] = numpy.roll(m[:, i], int(shift[i]))
#     return m


def stretch(m):
    factor = [1 + random.uniform(0., .5), 1 + random.uniform(0., .5)]
    target = tuple(int(f * s) for f, s in zip(factor, m[0].shape))
    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    out = (resize(n, target, preserve_range=True, mode='constant') for n in m)
    shift = [random.randint(0, s - 512) for s in target]

    return [o[shift[0]:shift[0] + 512, shift[1]: shift[1] + 512] for o in out]


def frankenstein(m):
    ops = [rot0, rot90, rot180, rot270, stretch, mirror]
    op = ops[random.randint(0, len(ops) - 1)]
    transforms = [lambda x:x, distord] # distord
    return transforms[random.randint(0,1)](op(m))


def batch_generator(it, batch_size=10, repeat=1, tocategorical=True):
    l = len(it) * repeat
    it = itertools.cycle(it)

    # Scale
    it = map(lambda m: (m[0] / numpy.iinfo(m[0].dtype).max, m[1].astype(int)), it)

    # Now augment
    it = map(frankenstein, it)

    # Build a steady dataset
    dataset = [next(it) for _ in range(l)]

    if batch_size == -1:
        batch_size = len(dataset)

    insider = itertools.cycle(dataset)

    while True:
        image, mask = list(zip(*[next(insider) for _ in range(batch_size)]))
        yield numpy.stack(image, axis=0)[..., numpy.newaxis], to_categorical(numpy.stack(mask, axis=0)) if tocategorical else numpy.stack(image, axis=0)
