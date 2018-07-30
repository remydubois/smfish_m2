from Segmentation.unet import unet_output
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Reshape
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from Segmentation.tensorboard_callback import MyTensorBoard
from os import listdir
import pickle
import re
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from Segmentation.feed_func import *

parser = ArgumentParser(description="Parser for reading target directory.")


def main(args_):
    # Read in:
    images = numpy.load('/Users/remydubois/Desktop/Remy/_REMY/Segmentation/data/mips.npy')
    masks = numpy.load('/Users/remydubois/Desktop/Remy/_REMY/Segmentation/data/masks.npy')

    input = Input((512, 512, 1))
    output = unet_output(input)

    X_train, X_test, y_train, y_test = train_test_split(images, masks, train_size=0.8)
    unet = Model(input, output)
    unet.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['acc'])
    tb = MyTensorBoard(log_dir='.', histogram_freq=0, write_batch_performance=True)
    # Checkpoint
    checkpointer = ModelCheckpoint(filepath='model-ckpt', verbose=0, save_best_only=True)

    batch_size = 5

    unet.fit_generator(batch_generator(X_train, y_train, batch_size=batch_size),
                       steps_per_epoch=X_train.shape[0] // batch_size,
                       callbacks=[tb, checkpointer],
                       epochs=5,
                       validation_data=batch_generator(X_test, y_test, batch_size=batch_size),
                       validation_steps=3,
                       verbose=1)



if __name__ == '__main__':
    args_ = parser.parse_args()

    main(args_)
