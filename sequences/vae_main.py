from keras.models import Model
from keras.optimizers import adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import sys
import os
from utils_vae import *
from callbacks import MyTensorBoard
from shutil import copyfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if '/Users/remydubois/anaconda3/lib/python3.6' in sys.path:
    LOCAL_TARGET = '/Users/remydubois/Dropbox/Remy/results/'
    LOCAL_SOURCE = '/Users/remydubois/Desktop/Remy/_REMY/Segmentation/data/'
else:
    LOCAL_TARGET = '/cbio/donnees/rdubois/results/'
    LOCAL_SOURCE = '/mnt/data40T_v2/rdubois/data/sequences/'


parser = argparse.ArgumentParser(description='Train vae.')
parser.add_argument('--epochs',
                    type=int,
                    default=5
                    )
parser.add_argument('--batch_size',
                    type=int,
                    default=10
                    )
parser.add_argument('--dataset',
                    default='mix.npy'
                    )
parser.add_argument('--logdir',
                    default='vae/'
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
parser.add_argument('--latent_dim',
                    type=int,
                    default=32
                    )


def kl_loss(x, z_decoded):
    # xent_loss = binary_crossentropy(K.flatten(x), K.flatten(z_decoded))
    kl_loss = - 0.5 * K.sum(1 + z_log_sigma -
                            K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    # return K.mean(kl_loss + xent_loss * (256 * 256))
    # reconstruction_error = xent_loss * 128 * 128
    # return K.mean(reconstruction_error + kl_loss)
    return kl_loss


if __name__ == '__main__':
    args = parser.parse_args()
    
    # Save code file
    if not os.path.exists(LOCAL_TARGET + args.logdir):
        os.mkdir(LOCAL_TARGET + args.logdir)
    copyfile('utils_vae.py', LOCAL_TARGET + args.logdir + '/code.py')

    
    os.environ["CUDA_VISIBLE_DEVICES"] = (args.gpu)


    # Get data
    clean = numpy.load(LOCAL_SOURCE + args.dataset)
    train_set, test_set = train_test_split(
        clean[:, 0], train_size=0.8, test_size=0.2)
    train = batch_generator_vae(
        train_set, args.latent_dim, batch_size=args.batch_size, repeat=args.repeat)
    test = batch_generator_vae(
        test_set, args.latent_dim, batch_size=args.batch_size, repeat=args.repeat)
    
    # Define model
    shape = next(train)[0][0].shape
    original_input_ = Input(shape)
    image_encoded, z_mean, z_log_sigma = encoder_output_sq3(original_input_, args.latent_dim, args.batch_size)
    encoder = Model(original_input_, [image_encoded, z_mean, z_log_sigma], name='encoder')

    encoded_input_ = Input((args.latent_dim,))
    image_decoded = decoder_output_sq3(encoded_input_)
    decoder = Model(encoded_input_, image_decoded, name='decoder')

    # Second input is just to authorize keras to compute the two losses independently. ideally I would just give it a mock tensor of the right shape
    vae = Model(original_input_, [decoder(encoder(original_input_)[0]), encoder(original_input_)[1]])

    # Compile
    vae.compile(optimizer='adam', loss=[binary_crossentropy, kl_loss], loss_weights=[shape[0] * shape[1], 0.99])

    # Misc
    earl = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
    mck = ModelCheckpoint(filepath=LOCAL_TARGET + args.logdir +
                          '/model-ckpt', verbose=0, save_best_only=True)
    tb = MyTensorBoard(log_dir=LOCAL_TARGET + args.logdir, histogram_freq=0, batch_size=10, write_batch_performance=True)


    # Now fit
    print("Init loss:", vae.evaluate_generator(
        test, steps=test_set.shape[0] // args.batch_size))
    try:
        vae.fit_generator(train,
                          steps_per_epoch=args.repeat * train_set.shape[0] // args.batch_size,
                          validation_data=test,
                          validation_steps=args.repeat *
                          test_set.shape[0] // args.batch_size,
                          epochs=args.epochs,
                          callbacks=[earl, mck, tb]
                          )
    except (KeyboardInterrupt, SystemExit):
        pass


    # Save architecture
    js = vae.to_json()
    with open(LOCAL_TARGET + args.logdir + '/architecture_vae.json', 'w') as fout:
    	fout.write(js)
    js = decoder.to_json()
    with open(LOCAL_TARGET + args.logdir + '/architecture_decoder.json', 'w') as fout:
    	fout.write(js)

    
    # predict a few samples
    if args.predict:
        seed, _ = next(test)
        vae.load_weights(LOCAL_TARGET + args.logdir + '/model-ckpt')
        pred = vae.predict(seed, verbose=1)[0]
        numpy.save(LOCAL_TARGET + args.logdir + '/preds.npy', numpy.concatenate([seed, pred], axis=-1))

    # generate a few samples
    if args.generate:
        seed = numpy.random.normal(size=(20, args.latent_dim))
        vae.load_weights(LOCAL_TARGET + args.logdir + '/model-ckpt')
        gen = decoder.predict(seed, verbose=1)
        numpy.save(LOCAL_TARGET + args.logdir + '/generated.npy', gen)

    
