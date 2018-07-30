from feed_func import *
from unet import *
from keras.layers import Input
from tensorboard_callback import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
import shutil
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Reshape
import argparse
from keras.losses import categorical_crossentropy, binary_crossentropy
import sys
import os
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if '/Users/remydubois/anaconda3/lib/python3.6' in sys.path:
    LOCAL_TARGET = '/Users/remydubois/Dropbox/Remy/results/'
    LOCAL_SOURCE = '/Users/remydubois/Desktop/segmented/'
else:
    LOCAL_TARGET = '/cbio/donnees/rdubois/results/'
    LOCAL_SOURCE = '/mnt/data40T_v2/rdubois/data/segmented/'

parser = argparse.ArgumentParser(description='Train vae.')
parser.add_argument('--epochs',
                    type=int,
                    default=10
                    )
parser.add_argument('--batch_size',
                    type=int,
                    default=10
                    )
parser.add_argument('--logdir',
                    default='unet/'
                    )
parser.add_argument('--predict',
                    type=int,
                    default=1
                    )
parser.add_argument('--generate',
                    type=int,
                    default=1
                    )
parser.add_argument('--repeat',
                    type=int,
                    default=10
                    )
parser.add_argument('--gpu',
                    default="0"
                    )
parser.add_argument('--weights',
                    default="1,1,1"
                    )
parser.add_argument('--pretrain',
                    type=int,
                    default=0
                    )


def jaccard_distance_loss(y_true, y_pred):
    smooth = 100
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def unet_loss(weights):
    
    def loss(y_true, y_pred):
        """
        TODO why not simply tmp = (y_true * y_pred); tmp[...,i] *= weight[i]; return -tf.reduce_sum(tmp, ) with clip etc dont mess with the acis in reduce sum. 
        check source below
        https://github.com/keras-team/keras/blob/2d183db0372e5ac2a686608cb9da0a9bd4319764/keras/backend/tensorflow_backend.py#L3176
        """

        weight_tensor = K.ones_like(y_pred)
        # subweight = numpy.array(weights).reshape(1, 1, 1, -1)
        subweight = weights / K.sum(weights)
        weight_tensor *= subweight

        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        
        prod = y_true * tf.log(y_pred)
        
        prod *= weight_tensor

        return - tf.reduce_sum(prod, -1)

    return loss

if __name__ == '__main__':
    args = parser.parse_args()
    weights = numpy.array(tuple(map(float, args.weights.split(',')))).reshape(1, 1, 1, -1)
    # print(weights)
    W = K.variable(numpy.ones_like(weights), dtype='float32', name='weights')
    
    def loss(y_true, y_pred):   
        """
        TODO why not simply tmp = (y_true * y_pred); tmp[...,i] *= weight[i]; return -tf.reduce_sum(tmp, ) with clip etc dont mess with the acis in reduce sum. 
        check source below
        https://github.com/keras-team/keras/blob/2d183db0372e5ac2a686608cb9da0a9bd4319764/keras/backend/tensorflow_backend.py#L3176
        """

        weight_tensor = K.ones_like(y_pred)
        # subweight = numpy.array(weights).reshape(1, 1, 1, -1)
        subweight = W / K.sum(W)
        weight_tensor *= subweight

        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        
        prod = y_true * tf.log(y_pred)
        
        prod *= weight_tensor

        return - tf.reduce_sum(prod, -1)
    
    logdir = 'UNET/unet_' + args.weights # + '_jac/'
    
    if not os.path.exists(LOCAL_TARGET + logdir):
        os.mkdir(LOCAL_TARGET + logdir)
    else:
        shutil.rmtree(LOCAL_TARGET + logdir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    stack = numpy.load(LOCAL_SOURCE + 'cm_IB_dapi_conf2.npy')
    data = stack[:, :2, :, :]
    if weights.shape[-1] > 3:
        data[:, 1, ...][stack[:, 2, ...]==1] = 3
    if weights.shape[-1] < 3:
        data[:, 1, ...] = stack[:, 2, ...]
    # if weights.shape[-1] == 3:
    #     data[:, 1, ...][data[:, 1, ...]==2] = 1
    #     data[:, 1, ...][stack[:, 2, ...]==1] = 2
    
    train_set, test_set = train_test_split(data, train_size=0.8, test_size=0.2)
    train_generator = batch_generator(train_set, batch_size=args.batch_size, repeat=args.repeat)
    Xtest = test_set[:,0, :, :, numpy.newaxis] / numpy.iinfo(test_set[:, 0].dtype).max
    ytest = to_categorical(test_set[:, 1].astype(int))
    n_steps_train = (args.repeat * train_set.shape[0]) // args.batch_size

    input_ = Input((512, 512, 1))
    if args.pretrain:
        # unet.load_weights(LOCAL_TARGET + 'UNET/pretrain_complete/model-ckpt')
        pretrained = Model(input_, unet_ae(input_))
        pretrained.load_weights(LOCAL_TARGET + 'UNET/pretrain_complete/model-ckpt')
        conv9 = Conv2D(weights.shape[-1], 3, activation='relu', padding='same', kernel_initializer='glorot_normal', name='conv23')(pretrained.layers[-4].output)
        output_ = Conv2D(weights.shape[-1], 1, activation='softmax', name='conv24')(conv9)
        unet = Model(input_, output_)
    else:
        output_ = unet_output(input_, n_classes=weights.shape[-1])
        unet = Model(input_, output_, name='unet')
    
    unet.compile(
        optimizer='adam', 
        loss=loss, 
        metrics=['acc'])

    earl = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
    mck = ModelCheckpoint(filepath=LOCAL_TARGET + logdir +
                                   'model-ckpt', verbose=0, save_best_only=True)
    tb = MyTB2(log_dir=LOCAL_TARGET + logdir, save_predictions=True, histogram_freq=0, write_graph=False, extra=W[0, 0, 0, 2] if weights.shape[-1] > 2 else None)
    red = ReduceLROnPlateau('val_loss', factor=0.5, patience=5)
    cbs = [mck, tb, red]
    if weights.shape[-1] > 2:
        vc = VarChanger(W, scale=weights, loc=15 * n_steps_train)
        cbs += [vc]
    # # Build class weights:
    # class_weight = numpy.ones((shape), dtype=None, order='C')
    # Now train
    # print("Init loss:", unet.evaluate(Xtest, ytest))
    try:
        unet.fit_generator(
            train_generator,
            # class_weight={0:1, 1:1, 2:10},
            steps_per_epoch=n_steps_train,
            validation_data=(Xtest, ytest),
            epochs=args.epochs,
            callbacks=cbs
        )

    except (KeyboardInterrupt, SystemExit):
        pass
