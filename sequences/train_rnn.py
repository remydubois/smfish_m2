from seq_rnn import *
from keras.layers import GRU, Input, GRUCell, Dropout, Dense, LSTM, CuDNNGRU, ConvLSTM2D, BatchNormalization
from keras.models import Model
from keras.optimizers import adagrad
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ("0")

if '/Users/remydubois/anaconda3/lib/python3.6' in sys.path:
    LOCAL_TARGET = '/Users/remydubois/Dropbox/Remy/results/'
    LOCAL_SOURCE = '/Users/remydubois/Desktop/Remy/_REMY/Segmentation/data/'
else:
    LOCAL_TARGET = '/cbio/donnees/rdubois/results/'
    LOCAL_SOURCE = '/mnt/data40T_v2/rdubois/data/sequences/'



parser = argparse.ArgumentParser(description='Train rnn.')
parser.add_argument('--epochs',
	type=int,
	default=5
	)
parser.add_argument('--batch_size',
	type=int,
	default=50
	)
parser.add_argument('--logdir',
	default='rnnseq/'
	)
parser.add_argument('--eval_freq',
	type=int,
	default=1
	)
parser.add_argument('--seq_len',
	type=int,
	default=201
	)
parser.add_argument('--pretrain',
	type=int,
	default=0
	)
parser.add_argument('--dataset',
	default='mix.npy'
	)

if __name__ == '__main__':
	args = parser.parse_args()

	# Get data
	clean = numpy.load(LOCAL_SOURCE + args.dataset)
	train_set, test_set = train_test_split(list(zip(*clean))[0], train_size=0.8)

	
	# Define Model
	input_ = Input(shape=(args.seq_len, 2))
	# x = CuDNNGRU(128, return_sequences=True)(input_)
	# x = Dropout(0.4)(x)
	x = CuDNNGRU(512, return_sequences=True)(input_)
	x = Dropout(0.4)(x)
	x = CuDNNGRU(256, return_sequences=True)(x)
	x = Dropout(0.4)(x)
	x = CuDNNGRU(128)(x)
	x = Dropout(0.4)(x)
	x = Dense(64, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dense(2)(x)
	model = Model(input_, x)


	# Pretrain
	if args.pretrain:
		try:
			model.load_weights(LOCAL_TARGET + 'pretrain_rnn/' + 'model-ckpt')
		except:
			pass


	# Compile
	ada = adagrad()
	model.compile(optimizer=ada, loss='mean_squared_error')
	earl = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
	if not os.path.exists(LOCAL_TARGET + args.logdir):
		os.mkdir(LOCAL_TARGET + args.logdir)
	mck = ModelCheckpoint(filepath=LOCAL_TARGET + args.logdir + 'model-ckpt', verbose=0, save_best_only=True)


	# Now fit
	train = batch_generator(train_set, seq_len=args.seq_len)
	test = batch_generator(test_set, seq_len=args.seq_len)
	print("Init loss:", model.evaluate_generator(test, steps=10600 // args.batch_size))
	model.fit_generator(train,
	                    steps_per_epoch=43000 // (args.batch_size * args.eval_freq), 
	                    validation_data=test,
	                    validation_steps=10600 // args.batch_size,
	                    epochs=args.epochs,
	                    callbacks=[earl, mck]
	                    )


	# now pred a few sequences and their seed
	seed = next(test)[0]
	predictions_full = predict_sequence(model, seed, steps=200)
	predictions_50 = predict_sequence(model, seed, steps=50)

	numpy.save(LOCAL_TARGET + args.logdir + 'predictions_full.npy', predictions_full)
	numpy.save(LOCAL_TARGET + args.logdir + 'predictions_50.npy', predictions_50)
	numpy.save(LOCAL_TARGET + args.logdir + 'seed.npy', seed)

	js = model.to_json()
	with open(LOCAL_TARGET + args.logdir + 'architecture.json', 'w') as fout:
		fout.write(js)
