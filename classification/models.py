from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate, GRU, Input, BatchNormalization, \
    AveragePooling2D, Activation, Lambda, GlobalAveragePooling2D, Add, UpSampling2D
from keras.models import Model
import numpy
from keras import objectives
import tensorflow as tf
from keras.backend import squeeze
from keras.applications.inception_v3 import InceptionV3


def fire_module(id, squeeze, expand):
    """
    Fire module, as in paper.
    :param input_, id, squeeze, expand: input tensor (object).
    :return: as in paper.
    """

    def layer(input_):
        with tf.name_scope('fire_module_%i' % id):
            conv_squeezed = Conv2D(squeeze, (1, 1), padding='valid', name='fm_%i_s1x1' % id, activation='relu')(input_)

            left = Conv2D(expand, (1, 1), padding='valid', name='fm_%i_e1x1' % id, activation='relu')(conv_squeezed)

            right = Conv2D(expand, (3, 3), padding='same', name='fm_%i_e3x3' % id, activation='relu')(conv_squeezed)

            out = concatenate([left, right], axis=-1, name='fire_module_%i' % id + 'concat')

            return out

    return layer


def SqueezeNetOutput(input_, num_classes=4, bypass=None):
    valid = [None, 'simple', 'complex']
    if bypass not in valid:
        raise UserWarning('"bypass" argument must be one of %s.' % ', '.join(map(str, valid)))

    conv_0 = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv_0', activation='relu')(input_)
    mxp_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_0')(conv_0)

    # Block 1
    fm_2 = fire_module(id=2, squeeze=16, expand=64)(mxp_0)
    fm_3 = fire_module(id=3, squeeze=16, expand=64)(fm_2)
    input_fm_4_ = fm_3
    if bypass == 'simple':
        input_fm_4_ = Add()([fm_2, fm_3])
    fm_4 = fire_module(id=4, squeeze=32, expand=128)(input_fm_4_)
    mxp_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_1')(fm_4)

    # Block 2
    fm_5 = fire_module(id=5, squeeze=32, expand=128)(mxp_1)
    input_fm_6_ = fm_5
    if bypass == 'simple':
        input_fm_6_ = Add()([mxp_1, fm_5])
    fm_6 = fire_module(id=6, squeeze=48, expand=192)(input_fm_6_)
    fm_7 = fire_module(id=7, squeeze=48, expand=192)(fm_6)
    input_fm_8_ = fm_7
    if bypass == 'simple':
        input_fm_8_ = Add()([fm_6, fm_7])
    fm_8 = fire_module(id=8, squeeze=64, expand=256)(input_fm_8_)
    mxp_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_2')(fm_8)

    # Block 3
    fm_9 = fire_module(id=9, squeeze=64, expand=256)(mxp_2)
    input_conv_10_ = fm_9
    if bypass == 'simple':
        input_conv_10_ = Add()([mxp_2, fm_9])
    # embedding = GlobalAveragePooling2D(name='embedding_layer')(input_conv_10_)
    dropped = Dropout(0.5, name='Dropout')(input_conv_10_)
    conv_10 = Conv2D(num_classes, (1, 1), padding='valid', name='conv10', activation='relu')(dropped)
    normalized = BatchNormalization(name='batch_normalization')(conv_10)

    # Predictions
    avgp_0 = GlobalAveragePooling2D(name='globalaveragepooling')(normalized)
    probas = Activation('softmax', name='probabilities')(avgp_0)

    return probas


def SqueezeNetOutputDA(input_, num_classes=4, lam=0.75, bypass=None):
    valid = [None, 'simple', 'complex']
    if bypass not in valid:
        raise UserWarning('"bypass" argument must be one of %s.' % ', '.join(map(str, valid)))

    conv_0 = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv_0', activation='relu')(input_)
    mxp_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_0')(conv_0)

    # Block 1
    fm_2 = fire_module(id=2, squeeze=16, expand=64)(mxp_0)
    fm_3 = fire_module(id=3, squeeze=16, expand=64)(fm_2)
    input_fm_4_ = fm_3
    if bypass == 'simple':
        input_fm_4_ = Add()([fm_2, fm_3])
    fm_4 = fire_module(id=4, squeeze=32, expand=128)(input_fm_4_)
    mxp_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_1')(fm_4)

    # Block 2
    fm_5 = fire_module(id=5, squeeze=32, expand=128)(mxp_1)
    input_fm_6_ = fm_5
    if bypass == 'simple':
        input_fm_6_ = Add()([mxp_1, fm_5])
    fm_6 = fire_module(id=6, squeeze=48, expand=192)(input_fm_6_)
    fm_7 = fire_module(id=7, squeeze=48, expand=192)(fm_6)
    input_fm_8_ = fm_7
    if bypass == 'simple':
        input_fm_8_ = Add()([fm_6, fm_7])
    fm_8 = fire_module(id=8, squeeze=64, expand=256)(input_fm_8_)
    mxp_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_2')(fm_8)

    # Block 3
    fm_9 = fire_module(id=9, squeeze=64, expand=256)(mxp_2)
    input_conv_10_ = fm_9
    if bypass == 'simple':
        input_conv_10_ = Add()([mxp_2, fm_9])
    dropped = Dropout(0.5, name='Dropout_pattern')(input_conv_10_)
    conv_10 = Conv2D(num_classes, (1, 1), padding='valid', name='conv10', activation='relu')(dropped)
    normalized = BatchNormalization(name='batch_normalization')(conv_10)

    # Predictions
    embedding = GlobalAveragePooling2D(name='embedding_layer')(input_conv_10_)
    avgp_0 = GlobalAveragePooling2D(name='globalaveragepooling')(normalized)
    probas = Activation('softmax', name='pattern_classifier')(avgp_0)

    
    reversed = Lambda(lambda t: (-lam) * t + tf.stop_gradient((1 + lam) * t), name='GRL')(embedding)
    dense_1 = Dense(128, activation='relu', name='dense_domain')(reversed)
    dense_1_bis = Dropout(0.5, name='Dropout_domain')(dense_1)
    probas_domain = Dense(2, activation='softmax', name='domain_classifier')(dense_1_bis)
    return probas, probas_domain


def sqn_fex(input_, bypass=None):
    valid = [None, 'simple', 'complex']
    if bypass not in valid:
        raise UserWarning('"bypass" argument must be one of %s.' % ', '.join(map(str, valid)))

    conv_0 = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv_0', activation='relu')(input_)
    mxp_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_0')(conv_0)

    # Block 1
    fm_2 = fire_module(id=2, squeeze=16, expand=64)(mxp_0)
    fm_3 = fire_module(id=3, squeeze=16, expand=64)(fm_2)
    input_fm_4_ = fm_3
    if bypass == 'simple':
        input_fm_4_ = Add()([fm_2, fm_3])
    fm_4 = fire_module(id=4, squeeze=32, expand=128)(input_fm_4_)
    mxp_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_1')(fm_4)

    # Block 2
    fm_5 = fire_module(id=5, squeeze=32, expand=128)(mxp_1)
    input_fm_6_ = fm_5
    if bypass == 'simple':
        input_fm_6_ = Add()([mxp_1, fm_5])
    fm_6 = fire_module(id=6, squeeze=48, expand=192)(input_fm_6_)
    fm_7 = fire_module(id=7, squeeze=48, expand=192)(fm_6)
    input_fm_8_ = fm_7
    if bypass == 'simple':
        input_fm_8_ = Add()([fm_6, fm_7])
    fm_8 = fire_module(id=8, squeeze=64, expand=256)(input_fm_8_)
    mxp_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_2')(fm_8)

    # Block 3
    fm_9 = fire_module(id=9, squeeze=64, expand=256)(mxp_2)
    input_conv_10_ = fm_9
    if bypass == 'simple':
        input_conv_10_ = Add()([mxp_2, fm_9])

    embedding = GlobalAveragePooling2D(name='embedding_layer')(input_fm_8_)

    return embedding

def pattern_classifier(x, l=1.):
    x = Lambda(lambda t: l * t + tf.stop_gradient((1 - l) * t), name='Scaler')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(6, activation='softmax', name='pattern_classifier')(x)

    return x

def smf_feature_ex(input_):
    
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_)
    x = MaxPooling2D((2,2))(x)
    # x = BatchNormalization()(x)
    
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Flatten(name='embedding_layer')(x)
    
    return x

def InceptionOutput(inputs, num_classes=5, pretrain=True):
    if pretrain:
        weights='imagenet'
    else:
        weights=None
    base_model = InceptionV3(input_tensor=inputs, include_top=False, weights=weights)

    base_output_ = base_model.output
    avgp = GlobalAveragePooling2D(name='GlobalAveragePooling')(base_output_)

    dense_1 = Dense(1024, activation='relu')(avgp)

    predictions = Dense(num_classes, activation='softmax')(dense_1)

    return predictions


def AE(inputs):
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_0')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_1')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = MaxPooling2D((3, 3), padding='same')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same', name='conv_4')(x)
    encoded = MaxPooling2D((5, 5), padding='same')(x)

    # encoded tensor shape is (5, 5, 4)

    x = Conv2D(4, (3, 3), activation='relu', padding='same', name='conv_5')(encoded)
    x = UpSampling2D((5, 5))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_6')(x)
    x = UpSampling2D((3, 3))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_7')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_8')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_9')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='conv_10')(x)

    return decoded