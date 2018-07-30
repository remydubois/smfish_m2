from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Reshape, concatenate, Lambda
from argparse import ArgumentParser
import keras.backend as K
import tensorflow as tf

def depthwise_softmax(x):

    exp_tensor = K.exp(x - K.max(x, axis=-1, keepdims=True))
    # softmax_tensor = exp_tensor / K.sum(exp_tensor, axis=-1, keepdims=True)

    return exp_tensor / K.sum(exp_tensor, axis=-1, keepdims=True)


def unet_output(inputs, n_classes=2):
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv1')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv3')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv4')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv5')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv6')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(drop3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv7')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv8')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv9')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv10')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv11')(
        UpSampling2D(size=(2, 2), name='up1')(drop5))
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv12')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv13')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv14')(
        UpSampling2D(size=(2, 2), name='up2')(conv6))
    merge7 = concatenate([conv3, up7], -1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv15')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv16')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv17')(
        UpSampling2D(size=(2, 2), name='up3')(conv7))
    merge8 = concatenate([conv2, up8], -1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv18')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv19')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv20')(
        UpSampling2D(size=(2, 2), name='up4')(conv8))
    merge9 = concatenate([conv1, up9], -1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv21')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv22')(conv9)
    conv9 = Conv2D(n_classes, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv23')(conv9)
    conv10 = Conv2D(n_classes, 1, activation='softmax', name='conv24')(conv9)
    
    return conv10


def unet_ae(inputs):
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv1')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv3')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv4')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv5')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv6')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(drop3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv7')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv8')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv9')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv10')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv11')(
        UpSampling2D(size=(2, 2), name='up1')(drop5))
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv12')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv13')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv14')(
        UpSampling2D(size=(2, 2), name='up2')(conv6))
    merge7 = concatenate([conv3, up7], -1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv15')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv16')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv17')(
        UpSampling2D(size=(2, 2), name='up3')(conv7))
    merge8 = concatenate([conv2, up8], -1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv18')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv19')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv20')(
        UpSampling2D(size=(2, 2), name='up4')(conv8))
    merge9 = concatenate([conv1, up9], -1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv21')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv22')(conv9)
    conv9 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv23')(conv9)
    conv10 = Conv2D(1, 1, name='conv24')(conv9)

    squeeze = Lambda(lambda x: K.squeeze(x, axis=-1))(conv10)

    return squeeze
