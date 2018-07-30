# Domain Adaptation proof of concept on MNIST
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Lambda, Flatten, Dropout, BatchNormalization, Lambda, Add, Reshape, UpSampling2D, Conv2DTranspose
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Model, Sequential
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy
import os
from keras.utils import to_categorical
import keras
import warnings
import keras.backend as K
from skimage.morphology import erosion, disk
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.utils import shuffle
import tqdm
from itertools import cycle
from keras.losses import categorical_crossentropy
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--logdir')
parser.add_argument('--blurr',
                    type=int,
                    default=0)
parser.add_argument('--da',
                    type=int,
                    default=1)
parser.add_argument('--freq',
                    type=int,
                    default=1)

# parser.add_argument('logdir',
# 					default='mist/')
# parser.add_argument('logdir',
# 					default='mist/')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if '/Users/remydubois/anaconda3/lib/python3.6' in sys.path:
    SOURCE = '/Users/remydubois/MNIST/'
    TARGET = '/Users/remydubois/Dropbox/Remy/results/'
else:
    SOURCE = '/mnt/data40T_v2/rdubois/data/MNIST/'
    TARGET = '/cbio/donnees/rdubois/results/'


def load_mnist():
    X = numpy.load(SOURCE + 'X.npy')
    y = numpy.load(SOURCE + 'y.npy')

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=12345)

    return Xtrain, Xtest, ytrain, ytest


def feature_ex(x):

    # x = Lambda(lambda t: K.spatial_2d_padding(t, padding=((2, 2), (2, 2))))(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    # x = BatchNormalization()(x)

    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Dropout(0.5)(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    x = Flatten(name='embedding_layer')(x)

    return x


def reconstructor(x, l=1.):

    x = Dense(256, activation='relu', name='HL_dense')(x)

    x = Reshape((4, 4, -1))(x)

    x = Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)

    x = Conv2DTranspose(1, (3, 3), padding='same',
                        activation='relu', name='reconstructed')(x)

    x = Lambda(lambda t: reverse_grad(t, l=-l),
               name='reconstruction_scaler')(x)

    return x


def digit_classifier(x, l=1.):
    x = Lambda(lambda t: reverse_grad(t, l=-l), name='Scaler')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='dense_digit')(x)
    x = Dense(10, activation='softmax', name='pattern_classifier')(x)

    return x


def reverse_grad(input_, l=1.):
    # t = - l * input_
    # reversed_gradient = t + tf.stop_gradient(input_ - t)
    return (-l) * input_ + tf.stop_gradient((1 + l) * input_)


def domain_classifier(x, l=1.):
    x = Lambda(lambda m: reverse_grad(m, l=l))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(2, activation='softmax', name='domain_classifier')(x)

    return x


def datagenerator(X, y1, y2=None, batch_size=50):

    if y2 is not None:
        assert y1.shape[0] == y2.shape[0]
        y2 = cycle(y2)
    X = cycle(X)
    y1 = cycle(y1)

    while True:

        xbatch = numpy.stack([next(X) for _ in range(batch_size)], axis=0)
        ybatch1 = numpy.stack([next(y1) for _ in range(batch_size)], axis=0)
        if y2 is not None:
            ybatch2 = numpy.stack([next(y2)
                                   for _ in range(batch_size)], axis=0)
            ybatch = [ybatch1, ybatch2]
        else:
            ybatch = ybatch1

        yield xbatch, ybatch


def patternloss(ytrue, ypred):
    # Count the number of synthetic elements (i.e. the ones where there is an
    # actual pattern class)
    retain = tf.greater(K.argmax(ytrue, axis=1), 0)

    # ytrue = tf.boolean_mask(ytrue, retain)
    # ypred = tf.boolean_mask(ypred, retain)
    # scale = 1 / K.mean(retain)

    ypred /= tf.reduce_sum(ypred, -1, True)
    _epsilon = tf.convert_to_tensor(1e-7, ypred.dtype.base_dtype)
    ypred = tf.clip_by_value(ypred, _epsilon, 1. - _epsilon)

    xent = - K.sum(ytrue * tf.log(ypred)) / \
        tf.count_nonzero(retain, dtype='float32')
    return xent  # * scale


def reconstructionloss(ytrue, ypred):
    """
    basically just a mse which excludes full zero images (synthetic). the mean has to be computed only on the real images (ie a subset of the batch)
    """

    # is zero if the image in synthetic (during training), therefore,
    # identifies images which should not be taken into account during the
    # training phase.
    retain = tf.reduce_max(ytrue, axis=(1, 2, 3), keepdims=False)
    mask = tf.greater(retain, 0)

    ytrue = tf.boolean_mask(ytrue, retain)
    ypred = tf.boolean_mask(ypred, retain)
    
    mse = K.mean(K.square(ytrue - ypred), axis=-1)

    return tf.reduce_mean(mse)


def domainloss(ytrue, ypred):
    ypred /= tf.reduce_sum(ypred, -1, True)
    _epsilon = tf.convert_to_tensor(1e-7, ypred.dtype.base_dtype)
    ypred = tf.clip_by_value(ypred, _epsilon, 1. - _epsilon)

    xent = - K.sum(ytrue * tf.log(ypred), axis=-1)

    return K.mean(xent)


def myacc(ytrue, ypred):
    """
    Must adapt metric to data from both domains (i.e. not count real data pattern prediction as it is meaningless)
    is messy but K.cast behaves very strangely, it seems to average / sum (?) when arguemnt is of boolean type. 
    """
    predictions = K.argmax(ypred, axis=1)
    truth = K.argmax(ytrue, axis=1)

    # contains 1 if synthetic (have access to label) and 0 if real (no acess
    # to label)
    retain = tf.greater(K.argmax(ytrue, axis=1), 0)

    equals = K.equal(truth, predictions)

    masked = tf.boolean_mask(equals, retain)

    # prod = K.cast(equals, 'float32') * retain

    # return K.sum(K.cast(equals, 'float32') * retain) / K.sum(retain)
    return K.cast(masked, 'float32')


def da(args):
    Xtrain, Xtest, ytrain, ytest = load_mnist()
    noise = numpy.load(SOURCE + 'noise.npy') * 3
    noise = numpy.repeat(noise, 2, axis=0)[:Xtrain.shape[0]]
    noise = shuffle(noise, random_state=0)

    # 1 for real images
    realtrain = numpy.random.randint(0, 2, size=Xtrain.shape[0]).astype(bool)
    realtest = numpy.random.randint(0, 2, size=Xtest.shape[0]).astype(bool)

    Xtest = Xtest + noise[:Xtest.shape[0]]
    Xtrain[realtrain] += noise[realtrain]
    ytrain[realtrain] *= 0
    Xtest[realtest] += noise[:Xtest.shape[0]][realtest]
    # This ensures that loss and acc are computed on real examples during
    # testing
    ytest[numpy.logical_not(realtest)] *= 0

    ytrain = [ytrain, to_categorical(realtrain.astype(int), num_classes=2)]
    ytest = [ytest, to_categorical(realtest.astype(int), num_classes=2)]

    train_generator = datagenerator(Xtrain, *ytrain)
    test_generator = datagenerator(Xtest, *ytest)

    input_ = Input((28, 28, 1))
    fex = feature_ex(input_)
    output_1 = digit_classifier(fex)
    output_2 = domain_classifier(fex)

    model = Model(input_, [output_1, output_2])
    # model = Model(input_, output_1)
    # model.summary()
    tb = TensorBoard(log_dir=TARGET + args.logdir)
    ck = ModelCheckpoint(TARGET + args.logdir + 'model-ckpt',
                         monitor='val_digit_classifier_loss')
    model.compile(optimizer='adam', loss=[patternloss, domainloss], metrics=[
                  myacc], loss_weights=[0.9, .1])

    freq = args.freq
    model.fit_generator(train_generator,
                        steps_per_epoch=(Xtrain.shape[0] // 50) // freq,
                        epochs=10 * freq,
                        callbacks=[tb, ck],
                        validation_data=test_generator,
                        validation_steps=(Xtest.shape[0] // 50),
                        )


# def noda():
# 	Xtrain, Xtest, ytrain, ytest = load_mnist()
# 	noise = numpy.load(SOURCE + 'noise.npy') * 4
# 	noise = numpy.repeat(noise, 2, axis=0)[:Xtest.shape[0]]
# 	noise = shuffle(noise, random_state=0)

# 	Xtest = Xtest + noise[:Xtest.shape[0]]
# 	# ytrain = [ytrain, to_categorical(real.astype(int), num_classes=2)]
# 	# ytest = [ytest, to_categorical([1,] * ytest.shape[0], num_classes=2)]


# 	train_generator = datagenerator(Xtrain, ytrain, y2=None)
# 	test_generator = datagenerator(Xtest, ytest, y2=None)

# 	input_ = Input((28, 28, 1))
# 	fex = feature_ex(input_)
# 	output_1 = digit_classifier(fex)

# 	model = Model(input_, output_1)

# 	tb = TensorBoard(log_dir=TARGET + args.logdir)
# 	ck = ModelCheckpoint(TARGET + args.logdir + 'model-ckpt', monitor='val_digit_classifier_loss')
# 	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# 	freq = 13
# 	model.fit_generator(train_generator,
# 		steps_per_epoch= (Xtrain.shape[0] // 50) // freq,
# 		epochs=10 * freq,
# 		callbacks=[tb, ck],
# 		validation_data=test_generator,
# 		validation_steps = (Xtest.shape[0] // 50),
# 		)

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    if args.da:
        da(args)
    else:
        noda()
