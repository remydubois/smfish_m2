import numpy
import matplotlib.pyplot as plt
from skimage.transform import resize
import itertools
import random
import tqdm
import math

def _flip(m, axis=0):
		cop = m.copy()
		cop[:, axis] = cop[:, axis].max() - cop[:, axis]
		return cop


def _transpose(m):
		return m[:, ::-1]


def _length(m):
		return numpy.sqrt(numpy.sum(numpy.square(m[1:] - m[:-1]), axis=1)).sum()


def _scale(m):
		l = _length(m)
		return 100 * (m.astype(float) / l)


def _stretch(m, ratios):
		assert min(ratios) > 0, "Ratios should be strictly positive."
		n = (m - numpy.mean(m, axis=0).astype(int)) * ratios
		# print(n, 'min', numpy.min(n, axis=0))
		return n - numpy.min(n, axis=0)


def _deconstruct(m):
		return m[1:] - m[:-1]


def _reconstruct(m):
		start = numpy.zeros((m.shape[0] + 1, m.shape[1]))
		progression = numpy.cumsum(numpy.vstack([[0, 0], m]), axis=0)
		return start + progression


def sample(m, n_points=200):
		out = resize(m, (n_points, 2), preserve_range=True, mode='constant', anti_aliasing=True)
		return out


def rot0(m):
		return m


def rot90(m):
		return _transpose(_flip(m, 0))


def rot180(m):
		return _flip(_flip(m, 1), 0)


def rot270(m):
		return _flip(_transpose(m), 0)


def gauss(x, loc=100, factor = 2, sigma=1):
		
		if type(loc) == float:
			loc = int(loc * x.shape[0])

		scale = 1 / math.sqrt(2 * math.pi * sigma ** 2)
		xs = numpy.arange(x.shape[0])
		y = numpy.roll(x, shift=x.shape[0]//2 - loc, axis=0)
		return x * (1 + (factor - 1) * scale * numpy.exp(-(xs - loc) ** 2/(2 * sigma ** 2))).reshape(-1, 1)


def frankenstein(x, n_ops=60):
		y = x.copy()
		
		for _ in range(n_ops):
			y = gauss(y, loc=random.uniform(0, 1), factor=random.uniform(0.2, 1.8), sigma=random.uniform(1, 8))

		return y 


def routine(x):
		# filter out small sequences (looking at histograms of length)
		ds = filter(lambda t: t.shape[0] > 200, x)
		
		# downsample sequences with a proper rescaling (not just integer slicing)
		ds = map(lambda m: sample(m, 201), ds)
		
		return ds


def cut_sequences(X, seq_len=201):
		return numpy.concatenate([numpy.roll(X, i, 1) for i in range(X.shape[1] - 1)], axis=0)[:, :seq_len, :]

		
def preprocess(x):
		# return map(lambda m: _deconstruct(_scale(m)), x)    
		return map(lambda m: _scale(m - numpy.mean(m, axis=0)), x)    


def augmentor(*ops):
		print
		def g(it):
				return map(lambda m: ops[random.randint(0, len(ops) - 1)](m), it)
		
		return g
				

def batch_generator(it, batch_size=30, seq_len=200):
		# only for pretraining
		it = map(frankenstein, it)
		it = routine(it)
		X = numpy.stack(list(it), axis=0)
		X = cut_sequences(X, seq_len + 1)

		numpy.random.shuffle(X)
		db = itertools.cycle(X)
		db = augmentor(rot0, rot90, rot180, rot270, lambda m: _stretch(m, [1.2, 1]), lambda m: _stretch(m, [1, 1.4]))(db)
		dbp = preprocess(db)

		while True:
				seqs = numpy.stack([dbp.__next__() for _ in range(batch_size)], axis=0)
				yield seqs[:, :seqs.shape[1] - 1, :], numpy.squeeze(seqs[:, seqs.shape[1] - 1, :])


def predict_sequence(model, seed, steps=200):
	seed = seed.copy()
	for i in tqdm.tqdm(range(steps), disable=False):
		next = model.predict(seed)
		seed = numpy.concatenate((seed, next[:, numpy.newaxis, ...]), 1)[:, 1:, :]
	# return [_reconstruct(s) for s in seed]
	return seed


def show_2(m):
		f, ax = plt.subplots(figsize=(10, 10))
		# m = sample(base)
		m = m.astype(int)
		n = m - numpy.min(m, axis=0)
		o = numpy.zeros(numpy.max(n, axis=0)+1)
		for i, t in enumerate(n):
				o[tuple(t)] = 200 + i
		ax.imshow(o)
		f.show()
