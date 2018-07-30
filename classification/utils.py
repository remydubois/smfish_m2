from keras.utils import to_categorical
from scipy.sparse import coo_matrix
from itertools import cycle
import random
from parameters import input_shape
from keras.losses import categorical_crossentropy
from tools import *
from Augmentation import *
import tensorflow as tf

import keras.backend as K


def _constitute_image(indices, shape=input_shape()):
    """
    Optimized for speedups in batch preparations. Approximately 250 microsec to yield the sparse matrix from indices.
    :param indices: numpy array of indices in the integer dtype (uint16)
    :param shape: output targetted shape
    :return: the sparse matrix
    """
    # transform it to the given shape. This is just done by translating the indices so that the given cell
    # is rather central in a shape-shaped image.
    # steps = ((numpy.array(shape) - numpy.max(indices, axis=1)) // 2).reshape(-1, 1)
    #
    # indices += steps
    data = [1, ] * indices.shape[1]
    out = coo_matrix((data, indices), shape=shape).toarray()

    return out


def _stack_batch(l):
    """
    Takes a batch of indices of type array([[xs],[ys],[channels]]) and return one array of kind
    array([[positions in batch], [xs], [ys], [channels]])
    :param l: batch of indices
    :return:
    """
    out = numpy.hstack([numpy.vstack(([idx, ] * m.shape[1], m)) for idx, m in enumerate(l)])
    return numpy.clip(out, 0, input_shape()[0] - 1) 


# @threadsafe_generator
def batch_generator_binary_images(dataset, batch_size, num_classes, channels=None, ae=False):
    """
    Vectorizing all the operations of image constitution. This version yield batches of 50 images in ~0.012s (with data
    augmentation) vs ~0.33s on the previous version. Performance is still linear in batch_size.
    :param dataset: the indices matrix
    :return: yields batches
    """
    
    global_generator = cycle(dataset)
    global_generator = augmentor(rot0, rot90, rot180, rot270)(global_generator)

    # with K.tf.device('/cpu:0'):
    while True:
        mask = numpy.zeros((batch_size, *input_shape(), 3))
        pre_batch, ys = list(zip(*[next(global_generator) for _ in range(batch_size)]))
        inds = _stack_batch(pre_batch)
        mask[tuple(inds)] = 1
        # if channels is not None:
        #     mask = mask[..., [channels]]

        if not ae:
            Y = to_categorical(ys, num_classes)
        else:
            Y = mask

        yield mask, Y


def datagenerator_DA(dataset, batch_size, num_classes, shape=input_shape()):

    cycled = cycle(dataset)
    global_generator = augmentor(rot0, rot90, rot180, rot270)(cycled)


    i = 0
    while True:
        i += 1
        pre_batch, yspattern, ysdomain = list(zip(*[next(global_generator) for _ in range(batch_size)]))
        
        mask = numpy.zeros((batch_size, *shape, 3))
        inds = _stack_batch(pre_batch)
        mask[tuple(inds)] = 1
        
        Ydomain = to_categorical(ysdomain, 2)
        
        Ypattern = to_categorical(yspattern, num_classes)
        Ypattern *= Ydomain[:, 0].reshape(-1, 1)  # ensures that the pattern loss is zero for synthetic examples.

        # print(i % (dataset.shape[0] // batch_size))
        if i == (dataset.shape[0] // batch_size + 1):
            #reinitialize
            sl = slice(0, min(batch_size, dataset.shape[0] - (i - 1) * batch_size))
            mask = mask[sl]
            Ypattern = Ypattern[sl]
            Ydomain = Ydomain[sl]

            cycled = cycle(dataset)
            global_generator = augmentor(rot0, rot90, rot180, rot270)(cycled)
            i = 0
            
        yield mask, [Ypattern, Ydomain]


def myacc(ytrue, ypred):
    """
    Must adapt metric to data from both domains (i.e. not count real data pattern prediction as it is meaningless)
    is messy but K.cast behaves very strangely, it seems to average / sum (?) when arguemnt is of boolean type. 
    """
    predictions = K.argmax(ypred, axis=1)
    truth = K.argmax(ytrue, axis=1)
    
    # contains 1 if synthetic (have access to label) and 0 if real (no acess to label)
    retain = tf.greater(K.argmax(ytrue, axis=1), 0)
    
    equals = K.equal(truth, predictions)

    masked = tf.boolean_mask(equals, retain)

    # prod = K.cast(equals, 'float32') * retain
    
    # return K.sum(K.cast(equals, 'float32') * retain) / K.sum(retain)
    return K.cast(masked, 'float32')


def synacc(ytrue, ypred):
    """
    Must adapt metric to data from both domains (i.e. not count real data pattern prediction as it is meaningless)
    is messy but K.cast behaves very strangely, it seems to average / sum (?) when arguemnt is of boolean type. 
    """
    predictions = K.argmax(ypred, axis=1)
    truth = K.argmax(ytrue[:,:-2], axis=1)
    
    # contains 1 if synthetic (have access to label) and 0 if real (no acess to label)
    retain = tf.less(K.argmax(ytrue[:, -2:], axis=1), 1)
    
    equals = K.equal(truth, predictions)

    masked = tf.boolean_mask(equals, retain)

    # prod = K.cast(equals, 'float32') * retain
    
    # return K.sum(K.cast(equals, 'float32') * retain) / K.sum(retain)
    return K.cast(masked, 'float32')

def realacc(ytrue, ypred):
    """
    Must adapt metric to data from both domains (i.e. not count real data pattern prediction as it is meaningless)
    is messy but K.cast behaves very strangely, it seems to average / sum (?) when arguemnt is of boolean type. 
    """
    predictions = K.argmax(ypred, axis=1)
    truth = K.argmax(ytrue[:,:-2], axis=1)
    
    # contains 1 if synthetic (have access to label) and 0 if real (no acess to label)
    retain = tf.greater(K.argmax(ytrue[:, -2:], axis=1), 0)
    
    equals = K.equal(truth, predictions)

    masked = tf.boolean_mask(equals, retain)

    # prod = K.cast(equals, 'float32') * retain
    
    # return K.sum(K.cast(equals, 'float32') * retain) / K.sum(retain)
    return K.cast(masked, 'float32')


def synpatternloss(ytrue, ypred):
    # Count the number of synthetic elements (i.e. the ones where there is an actual pattern class)
    retain = tf.less(K.argmax(ytrue[:, -2:], axis=1), 1)

    # ytrue = tf.boolean_mask(ytrue, retain)
    # ypred = tf.boolean_mask(ypred, retain)
    # scale = 1 / K.mean(retain)

    ypred /= tf.reduce_sum(ypred, -1, True)
    _epsilon = tf.convert_to_tensor(1e-7, ypred.dtype.base_dtype)
    ypred = tf.clip_by_value(ypred, _epsilon, 1. - _epsilon)

    xent = - K.sum(ytrue * tf.log(ypred)) / tf.count_nonzero(retain, dtype='float32')
    return xent # * scale

def realpatternloss(ytrue, ypred):
    # Count the number of synthetic elements (i.e. the ones where there is an actual pattern class)
    retain = tf.greater(K.argmax(ytrue[:, -2:], axis=1), 0)

    # ytrue = tf.boolean_mask(ytrue, retain)
    # ypred = tf.boolean_mask(ypred, retain)
    # scale = 1 / K.mean(retain)

    ypred /= tf.reduce_sum(ypred, -1, True)
    _epsilon = tf.convert_to_tensor(1e-7, ypred.dtype.base_dtype)
    ypred = tf.clip_by_value(ypred, _epsilon, 1. - _epsilon)

    xent = - K.sum(ytrue * tf.log(ypred)) / tf.count_nonzero(retain, dtype='float32')
    return xent # * scale


def patternloss(ytrue, ypred):
    # Count the number of synthetic elements (i.e. the ones where there is an actual pattern class)
    retain = tf.greater(K.argmax(ytrue, axis=1), 0)

    # ytrue = tf.boolean_mask(ytrue, retain)
    # ypred = tf.boolean_mask(ypred, retain)
    # scale = 1 / K.mean(retain)

    ypred /= tf.reduce_sum(ypred, -1, True)
    _epsilon = tf.convert_to_tensor(1e-7, ypred.dtype.base_dtype)
    ypred = tf.clip_by_value(ypred, _epsilon, 1. - _epsilon)

    xent = - K.sum(ytrue * tf.log(ypred)) / tf.count_nonzero(retain, dtype='float32')
    return xent # * scale


def domainloss(ytrue, ypred):
    ypred /= tf.reduce_sum(ypred, -1, True)
    _epsilon = tf.convert_to_tensor(1e-7, ypred.dtype.base_dtype)
    ypred = tf.clip_by_value(ypred, _epsilon, 1. - _epsilon)

    xent = - K.sum(ytrue * tf.log(ypred), axis=-1)

    return K.mean(xent)


# def domainloss(ytrue, ypred):
#     # Count the number of synthetic elements (i.e. the ones where there is an actual pattern class)
#     retain = K.max(ytrue, axis=-1)
#     # scale = 1 / (1 - K.mean(retain))

#     ypred /= tf.reduce_sum(ypred, -1, True)
#     _epsilon = tf.convert_to_tensor(1e-7, ypred.dtype.base_dtype)
#     ypred = tf.clip_by_value(ypred, _epsilon, 1. - _epsilon)

#     xent = - K.sum(ytrue * tf.log(ypred), axis=-1)
#     return xent * scale

