from keras.layers import GRU, Input, GRUCell, Dropout, Dense, LSTM, CuDNNGRU, ConvLSTM2D, BatchNormalization, Conv2D, GlobalMaxPooling2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Reshape, Flatten, Lambda, concatenate, Conv2DTranspose, Dropout, BatchNormalization, Add
import keras.backend as K
from keras import objectives
from keras.losses import binary_crossentropy
from seq_rnn import *
import tensorflow as tf
import keras.backend as K


def _stack_batch(l):
	"""
	Takes a batch of indices of type array([[xs],[ys],[channels]]) and return one array of kind
	array([[positions in batch], [xs], [ys], [channels]])
	:param l: batch of indices
	:return:
	"""
	return numpy.hstack([numpy.vstack(([idx, ] * m.shape[1], m, [0, ] * m.shape[1])) for idx, m in enumerate(l)])


def batch_generator_vae(it, LATENT_DIM, batch_size=30, repeat=10):
		# it = itertools.cycle(it)
		# it = augmentor(rot0, rot90, rot180, rot270)(it)
		l = len(it) * repeat
		it = map(lambda m: m - numpy.min(m, axis=0), it)
		# it = map(lambda m: frankenstein(m, 60), it)
		it = map(lambda m: (127 * m / m.max()).astype(int), it)
		it = itertools.cycle(map(numpy.transpose, it))
		dataset = [next(it) for _ in range(l)]
		random.shuffle(dataset)
		if batch_size == -1:
			batch_size = len(dataset)
		it = itertools.cycle(dataset)

		while True:
				mask = numpy.zeros((batch_size, 128, 128, 1))
				indices = _stack_batch([it.__next__() for _ in range(batch_size)])
				# print(indices)
				mask[tuple(indices)] = 1
				yield mask, [mask, numpy.zeros((batch_size, LATENT_DIM))]


def fire_module(id, expand, strides):
	"""
	Fire module, as in paper.
	:param input_, id, squeeze, expand: input tensor (object).
	:return: as in paper.
	"""

	def layer(input_):
		left = Conv2D(32, (3, 3), padding='same', activation='relu')(input_)
		right = Conv2D(32, (1, 1), padding='same', activation='relu')(input_)
		x = concatenate([left, right], axis=-1)
		x = Dropout(0.4)(x)

		return x

	return layer


def water_module(id, squeeze, expand, strides):
	"""
	Will output #squeeze number of channels
	"""


	def layer(input_):
		x = Conv2D(squeeze, (1, 1), padding='same', activation='relu')(input_)
		
		left = Lambda(lambda x: x[:, :, :, :squeeze // 2])(x)
		right = Lambda(lambda x: x[:, :, :, squeeze // 2:])(x)
		
		x = concatenate([
						Conv2DTranspose(expand // 2, (3, 3), padding='same', activation='relu')(left), 
						Conv2DTranspose(expand // 2, (1, 1), padding='same', activation='relu')(right)
						], axis=-1)
		
		x = Dropout(0.4)(x)

		return x

	return layer


def sampling(batch_size, LATENT_DIM):

	def gen(in_):
		z_mean, z_log_sigma = in_
		epsilon = K.random_normal(shape=(batch_size, LATENT_DIM))
		ex = K.exp(0.5 * z_log_sigma)
		return z_mean + ex * epsilon

	return gen


def encoder_output(original_input_, LATENT_DIM):

	# 1 -> 32
	x = Conv2D(32, (3, 3), padding='same', activation='relu')(original_input_)
	x = Dropout(0.2)(x)
	x = MaxPooling2D((2, 2))(x)

	# 32 -> 64
	left = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	x = MaxPooling2D((2, 2))(x)

	# 64 -> 128
	left = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	x = MaxPooling2D((2, 2))(x)

	# 128 -> 8
	left = Conv2D(4, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(4, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	x = MaxPooling2D((2, 2))(x)

	# Encoded state is (8, 8, 8)

	# 8 -> -1
	x = Flatten()(x)

	# #8x8 x 8 -> Latent Dim
	z_mean = Dense(LATENT_DIM)(x)
	z_log_sigma = Dense(LATENT_DIM)(x)
	x = Lambda(sampling)([z_mean, z_log_sigma])
	return x, z_mean, z_log_sigma


def decoder_output(latent_input_, LATENT_DIM):

	# Latent Dim -> 8x8 x 8
	x = Dense(8 * 8 * 8)(latent_input_)

	# (-1) -> 8. This is encoded the encoded image
	x = Reshape((8, 8, 8))(x)

	left = Lambda(lambda x: x[:, :, :, :4])(x)
	right = Lambda(lambda x: x[:, :, :, 4:])(x)
	x = concatenate([Conv2DTranspose(4, (3, 3), padding='same', activation='relu')(left), Conv2DTranspose(4, (1, 1), padding='same', activation='relu')(right)], axis=-1)
	x = Dropout(0.2)(x)
	x = UpSampling2D((2,2))(x)
	

	# 8 -> 128
	left = Lambda(lambda x: x[:, :, :, :4])(x)
	right = Lambda(lambda x: x[:, :, :, 4:])(x)
	x = concatenate([Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(left), Conv2DTranspose(64, (1, 1), padding='same', activation='relu')(right)], axis=-1)
	x = Dropout(0.2)(x)
	x = UpSampling2D((2, 2))(x)


	# 128 -> 64
	left = Lambda(lambda x: x[:, :, :, :64])(x)
	right = Lambda(lambda x: x[:, :, :, 64:])(x)
	x = concatenate([Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(left), Conv2DTranspose(32, (1, 1), padding='same', activation='relu')(right)], axis=-1)
	x = Dropout(0.2)(x)
	x = UpSampling2D((2, 2))(x)

	# 64 -> 32
	x = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.2)(x)
	x = UpSampling2D((2, 2))(x)

	# 32 -> 1
	x = Conv2DTranspose(1, (3, 3), padding='same',
			   activation='relu')(x)  # (256, 256, 1)

	return x


def encoder_output_sq(original_input_):

	# 1 -> 32
	x = Conv2D(32, (3, 3), strides=(2,2), padding='same', activation='relu')(original_input_)
	x = Dropout(0.2)(x)
	
	# 32 -> 64
	left = Conv2D(32, (3, 3), strides=(2,2), padding='same', activation='relu')(x)
	right = Conv2D(32, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	
	# 64 -> 64
	left = Conv2D(32, (3, 3), strides=(2,2), padding='same', activation='relu')(x)
	right = Conv2D(32, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	
	# 64 -> 128
	left = Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu')(x)
	right = Conv2D(64, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	
	# 128 -> 128
	# left = Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu')(x)
	# right = Conv2D(64, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	# x = concatenate([left, right], axis=-1)
	x = Conv2D(128, (1,1), strides=(2,2), padding='same', activation='relu')(x)
	x = Dropout(0.2)(x)
	
	# 128 -> 256
	left = Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='relu')(x)
	right = Conv2D(128, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	
	# Encoded state is (2, 2, 256)

	# 8 -> -1
	x = GlobalMaxPooling2D()(x)
	# x = Flatten()(x)

	# #8x8 x 8 -> Latent Dim
	z_mean = Dense(LATENT_DIM)(x)
	z_log_sigma = Dense(LATENT_DIM)(x)
	x = Lambda(sampling)([z_mean, z_log_sigma])
	return x, z_mean, z_log_sigma


def decoder_output_sq(latent_input_):

	# Latent Dim -> 8x8 x 8
	x = Dense(256)(latent_input_)

	# (-1) -> 8. This is encoded the encoded image
	x = Reshape((1, 1, 256))(x)
	x = Conv2DTranspose(256, (1, 1), strides=(2, 2), padding='same', activation='relu')(x)

	left = Lambda(lambda x: x[:, :, :, :128])(x)
	right = Lambda(lambda x: x[:, :, :, 128:])(x)
	x = concatenate([
					Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(left), 
					Conv2DTranspose(64, (1, 1), strides=(2, 2), padding='same', activation='relu')(right)
					], axis=-1)
	x = Dropout(0.2)(x)
	

	# left = Lambda(lambda x: x[:, :, :, :64])(x)
	# right = Lambda(lambda x: x[:, :, :, 64:])(x)
	# x = concatenate([
	# 				Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(left), 
	# 				Conv2DTranspose(64, (1, 1), strides=(2, 2), padding='same', activation='relu')(right)
	# 				], axis=-1)
	x = Conv2DTranspose(128, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	x = Dropout(0.2)(x)
	

	left = Lambda(lambda x: x[:, :, :, :64])(x)
	right = Lambda(lambda x: x[:, :, :, 64:])(x)
	x = concatenate([
					Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(left), 
					Conv2DTranspose(32, (1, 1), strides=(2, 2), padding='same', activation='relu')(right)
					], axis=-1)
	x = Dropout(0.2)(x)
	

	left = Lambda(lambda x: x[:, :, :, :32])(x)
	right = Lambda(lambda x: x[:, :, :, 32:])(x)
	x = concatenate([
					Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(left), 
					Conv2DTranspose(32, (1, 1), strides=(2, 2), padding='same', activation='relu')(right)
					], axis=-1)
	x = Dropout(0.2)(x)
	

	left = Lambda(lambda x: x[:, :, :, :32])(x)
	right = Lambda(lambda x: x[:, :, :, 32:])(x)
	x = concatenate([
					Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(left), 
					Conv2DTranspose(16, (1, 1), strides=(2, 2), padding='same', activation='relu')(right)
					], axis=-1)
	x = Dropout(0.2)(x)
	
	
	
	# 32 -> 1
	x = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # (256, 256, 1)

	return x


def encoder_output_sq2(original_input_):

	# 1 -> 32
	x = Dropout(0.4)(original_input_)
	x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
	
	# 32 -> 64
	x = Dropout(0.4)(x)
	left = Conv2D(32, (3, 3), strides=(2,2), padding='same', activation='relu')(x)
	right = Conv2D(32, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	
	# 64 -> 64
	x = Dropout(0.4)(x)
	left = Conv2D(32, (3, 3), strides=(2,2), padding='same', activation='relu')(x)
	right = Conv2D(32, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	
	# 64 -> 128
	x = Dropout(0.4)(x)
	left = Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu')(x)
	right = Conv2D(64, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	
	# 128 -> 256
	x = Dropout(0.4)(x)
	left = Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu')(x)
	right = Conv2D(64, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	
	# 256 -> 128
	x = Dropout(0.4)(x)
	left = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
	right = Conv2D(32, (1, 1), strides=(2, 2), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	
	# 256 -> 256
	# left = Conv2D(256, (3, 3), strides=(2,2), padding='same', activation='relu')(x)
	# right = Conv2D(256, (1, 1), strides=(2,2), padding='same', activation='relu')(x)
	# x = concatenate([left, right], axis=-1)
	# x = Dropout(0.4)(x)
	# Encoded state is (2, 2, 256)

	# 8 -> -1
	# x = GlobalMaxPooling2D()(x)
	x = Flatten()(x)

	# x = BatchNormalization()(x)
	x = Dropout(0.4)(x)
	x = Dense(512)(x)
	
	# #8x8 x 8 -> Latent Dim
	# x = Dropout(0.4)(x)
	z_mean = Dense(LATENT_DIM)(x)
	z_log_sigma = Dense(LATENT_DIM)(x)
	x = Lambda(sampling)([z_mean, z_log_sigma])
	return x, z_mean, z_log_sigma


def decoder_output_sq2(latent_input_):

	# Latent Dim -> 8x8 x 8
	x = Dropout(0.2)(latent_input_)
	x = Dense(512)(x)
	# x = BatchNormalization()(x)
	x = Dropout(0.4)(x)
	x = Dense(4 * 4 * 64)(x)
	
	# (-1) -> 8. This is encoded the encoded image
	x = Reshape((4, 4, 64))(x)
	# x = Conv2DTranspose(512, (1, 1), strides=(2,2), padding='same', activation='relu')(x)

	x = Dropout(0.4)(x)
	# x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
	left = Lambda(lambda x: x[:, :, :, :32])(x)
	right = Lambda(lambda x: x[:, :, :, 32:])(x)
	x = concatenate([
					Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(left), 
					Conv2DTranspose(64, (1, 1), strides=(2, 2), padding='same', activation='relu')(right)
					], axis=-1)
	# tmp = x
	
	x = Dropout(0.4)(x)
	# o = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
	# x = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
	# x = Add()([o, x])
	left = Lambda(lambda x: x[:, :, :, :64])(x)
	right = Lambda(lambda x: x[:, :, :, 64:])(x)
	x = concatenate([
					Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(left), 
					Conv2DTranspose(64, (1, 1), strides=(2, 2), padding='same', activation='relu')(right)
					], axis=-1)
	

	x = Dropout(0.4)(x)
	# x = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
	left = Lambda(lambda x: x[:, :, :, :64])(x)
	right = Lambda(lambda x: x[:, :, :, 64:])(x)
	x = concatenate([
					Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(left), 
					Conv2DTranspose(32, (1, 1), strides=(2, 2), padding='same', activation='relu')(right)
					], axis=-1)
	

	x = Dropout(0.4)(x)
	# x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
	left = Lambda(lambda x: x[:, :, :, :32])(x)
	right = Lambda(lambda x: x[:, :, :, 32:])(x)
	x = concatenate([
					Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(left), 
					Conv2DTranspose(32, (1, 1), strides=(2, 2), padding='same', activation='relu')(right)
					], axis=-1)
	
	x = Dropout(0.4)(x)
	# x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
	left = Lambda(lambda x: x[:, :, :, :32])(x)
	right = Lambda(lambda x: x[:, :, :, 32:])(x)
	x = concatenate([
					Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(left), 
					Conv2DTranspose(16, (1, 1), strides=(2, 2), padding='same', activation='relu')(right)
					], axis=-1)
	
	
	
	# 32 -> 1
	# x = Dropout(0.4)(x)
	x = Conv2DTranspose(1, (3, 3),	 padding='same', activation='relu')(x)  # (256, 256, 1)

	return x


def encoder_output_sq3(original_input_, LATENT_DIM, batch_size):

	# 1 -> 32
	left = Conv2D(32, (3, 3), padding='same', activation='relu')(original_input_)
	right = Conv2D(32, (1, 1), padding='same', activation='relu')(original_input_)
	x = concatenate([left, right], axis=-1)
	# x = Conv2D(64, (3, 3), padding='same', activation='relu')(original_input_)
	# x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
	x = MaxPooling2D((2, 2))(x)

	x = Dropout(0.2)(x)
	left = Conv2D(48, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(48, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	# x = Conv2D(96, (3, 3), padding='same', activation='relu')(x)
	# x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
	x = MaxPooling2D((2, 2))(x)

	x = Dropout(0.2)(x)
	left = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	# x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
	# x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
	x = MaxPooling2D((2, 2))(x)
	
	x = Dropout(0.2)(x)
	left = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	# x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
	x = MaxPooling2D((2, 2))(x)
	
	x = Dropout(0.2)(x)
	x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
	x = MaxPooling2D((2, 2))(x)
	
	x = Dropout(0.2)(x)
	x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
	x = MaxPooling2D((2, 2))(x)
		

	# Encoded state is (8, 8, 16)

	# 16 -> -1
	x = Flatten()(x)

	x = Dropout(0.1)(x)
	x = Dense(512)(x)

	# #8x8 x 16 -> Latent Dim
	z_mean = Dense(LATENT_DIM)(x)
	z_log_sigma = Dense(LATENT_DIM)(x)
	x = Lambda(sampling(batch_size, LATENT_DIM))([z_mean, z_log_sigma])
	return x, z_mean, z_log_sigma


def decoder_output_sq3(latent_input_):

	# Latent Dim -> 8x8 x 8
	x = Dense(512)(latent_input_)
	x = Dropout(0.1)(x)
	x = Dense(512)(x)

	# (-1) -> 8. This is encoded the encoded image
	x = Reshape((2, 2, 128))(x)
	
	x = Dropout(0.2)(x)
	left = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
	right = Conv2DTranspose(128, (1, 1), strides=(2, 2), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	
	x = Dropout(0.2)(x)
	x = Conv2DTranspose(192, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
	
	x = Dropout(0.1)(x)
	x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
	
	x = Dropout(0.1)(x)
	x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
	
	x = Dropout(0.1)(x)
	x = Conv2DTranspose(96, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
	
	x = Dropout(0.1)(x)
	x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # (128, 128, 1)

	x = Conv2D(1, (3, 3), padding='same', activation='relu')(x)  # (128, 128, 1)

	return x


def encoder_output_sq4(original_input_, LATENT_DIM):

	# 1 -> 32
	x = Conv2D(96, (3, 3), padding='same', activation='relu')(original_input_)
	x = Dropout(0.2)(x)
	x = MaxPooling2D((2, 2))(x)

	# 32 -> 64
	left = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	x = MaxPooling2D((2, 2))(x)
	
	# 64 -> 64
	left = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	
	# 64 -> 128
	left = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	x = MaxPooling2D((2, 2))(x)
	
	# 128 -> 128
	left = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	
	# 128 -> 64
	left = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	
	# 64 -> 16
	left = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
	right = Conv2D(8, (1, 1), padding='same', activation='relu')(x)
	x = concatenate([left, right], axis=-1)
	x = Dropout(0.2)(x)
	x = MaxPooling2D((2, 2))(x)

	# Encoded state is (8, 8, 16)

	# 16 -> -1
	x = Flatten()(x)

	# #8x8 x 16 -> Latent Dim
	z_mean = Dense(LATENT_DIM)(x)
	z_log_sigma = Dense(LATENT_DIM)(x)
	x = Lambda(sampling)([z_mean, z_log_sigma])
	return x, z_mean, z_log_sigma


def decoder_output_sq4(latent_input_, LATENT_DIM):

	# Latent Dim -> 8x8 x 8
	x = Dense(8 * 8 * 16)(latent_input_)
	x = BatchNormalization()(x)

	# (-1) -> 8. This is encoded the encoded image
	x = Reshape((8, 8, 16))(x)

	# 8 -> 16
	left = Lambda(lambda x: x[:, :, :, :8])(x)
	right = Lambda(lambda x: x[:, :, :, 8:])(x)
	x = concatenate([Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(left), Conv2DTranspose(32, (1, 1), padding='same', activation='relu')(right)], axis=-1)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	x = UpSampling2D((2,2))(x)
	

	# 16 -> 64
	left = Lambda(lambda x: x[:, :, :, :32])(x)
	right = Lambda(lambda x: x[:, :, :, 32:])(x)
	x = concatenate([Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(left), Conv2DTranspose(64, (1, 1), padding='same', activation='relu')(right)], axis=-1)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	

	# 64 -> 128
	left = Lambda(lambda x: x[:, :, :, :64])(x)
	right = Lambda(lambda x: x[:, :, :, 64:])(x)
	x = concatenate([Conv2DTranspose(256, (3, 3), padding='same', activation='relu')(left), Conv2DTranspose(256, (1, 1), padding='same', activation='relu')(right)], axis=-1)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	

	# 128 -> 128
	left = Lambda(lambda x: x[:, :, :, :256])(x)
	right = Lambda(lambda x: x[:, :, :, 256:])(x)
	x = concatenate([Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(left), Conv2DTranspose(128, (1, 1), padding='same', activation='relu')(right)], axis=-1)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	x = UpSampling2D((2, 2))(x)
	

	# 128 -> 64
	left = Lambda(lambda x: x[:, :, :, :128])(x)
	right = Lambda(lambda x: x[:, :, :, 128:])(x)
	x = concatenate([Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(left), Conv2DTranspose(64, (1, 1), padding='same', activation='relu')(right)], axis=-1)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	x = UpSampling2D((2,2))(x)
	
	# 64 -> 64
	left = Lambda(lambda x: x[:, :, :, :64])(x)
	right = Lambda(lambda x: x[:, :, :, 64:])(x)
	x = concatenate([Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(left), Conv2DTranspose(32, (1, 1), padding='same', activation='relu')(right)], axis=-1)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	
	# 64 -> 32
	x = Conv2DTranspose(96, (3, 3), padding='same', activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	x = UpSampling2D((2, 2))(x)

	# 32 -> 1
	x = Conv2DTranspose(1, (3, 3), padding='same',
			   activation='relu')(x)  # (256, 256, 1)

	return x


def encoder_output_sq5(original_input_, LATENT_DIM):

	# 1 -> 32
	x = Conv2D(32, (3, 3), padding='same', activation='relu')(original_input_)
	x = MaxPooling2D((2, 2))(x)
	
	x = fire_module(2, 16, 64)(x)
	x = MaxPooling2D((2, 2))(x)

	x = fire_module(5, 32, 128)(x)
	x = MaxPooling2D((2, 2))(x)


	x = fire_module(7, 48, 256)(x)
	x = MaxPooling2D((2, 2))(x)
	
	x = fire_module(9, 64, 512)(x)
	x = GlobalAveragePooling2D()(x)
	
	# x = Flatten()(x)

	# #8x8 x 8 -> Latent Dim
	z_mean = Dense(LATENT_DIM)(x)
	z_log_sigma = Dense(LATENT_DIM)(x)
	x = Lambda(sampling)([z_mean, z_log_sigma])
	return x, z_mean, z_log_sigma

def decoder_output_sq5(latent_input_, LATENT_DIM):

	x = Dense(512)(latent_input_)
	
	x = Reshape((1, 1, 512))(x)
	x = UpSampling2D((8, 8))(x)

	x = fire_module(10, 64, 512)(x)
	x = UpSampling2D((2, 2))(x)

	x = fire_module(8, 48, 256)(x)
	x = UpSampling2D((2, 2))(x)

	x = fire_module(6, 32, 128)(x)
	x = UpSampling2D((2, 2))(x)

	x = fire_module(4, 16, 64)(x)
	x = UpSampling2D((2, 2))(x)

	x = Conv2DTranspose(1, (3, 3), padding='same', activation='relu')(x)

	return x
	















