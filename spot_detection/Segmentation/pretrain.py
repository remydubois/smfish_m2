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


if __name__ == '__main__':
    args = parser.parse_args()

    logdir = 'UNET/unet_pretrain_hilo/'

    if not os.path.exists(LOCAL_TARGET + logdir):
        os.mkdir(LOCAL_TARGET + logdir)
    else:
        shutil.rmtree(LOCAL_TARGET + logdir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    stack = numpy.load(LOCAL_SOURCE + 'CM_mips_only_hilo.npy')
    data = numpy.stack([stack, ] * 2, axis=1)

    train_set, test_set = train_test_split(data, train_size=0.8, test_size=0.2)
    train_generator = batch_generator(
        train_set, batch_size=args.batch_size, repeat=args.repeat, tocategorical=False)
    Xtest = test_set[:, 0, :, :, numpy.newaxis] / \
        numpy.iinfo(test_set[:, 0].dtype).max
    ytest = numpy.squeeze(Xtest)
    n_steps_train = (args.repeat * train_set.shape[0]) // args.batch_size

    input_ = Input((512, 512, 1))
    output_ = unet_ae(input_)
    unet = Model(input_, output_, name='unet_ae')
    
    unet.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['acc'])

    earl = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
    mck = ModelCheckpoint(filepath=LOCAL_TARGET + logdir +
                          'model-ckpt', verbose=0, save_best_only=True)
    tb = MyTB2(log_dir=LOCAL_TARGET + logdir, save_predictions=False, histogram_freq=1,
               write_graph=False, extra=None, save_reconstructions=True)
    red = ReduceLROnPlateau('val_loss', factor=0.5, patience=5)
    cbs = [mck, tb, red]
    
    print("Init loss:", unet.evaluate(Xtest, ytest))
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
