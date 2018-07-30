import numpy
import pandas
import itertools
import os
import matplotlib.pyplot as plt
import tqdm
from sklearn.preprocessing import LabelEncoder

"""
One use script to turn the experimental data into a file in the same format as the 'merged files' from Preprocessing.
"""

DATA_PATH = '/Users/remydubois/Your team Dropbox/Rémy Dubois/Remy/Data/180612-Experimental/'

def parse_text(text):
	"""
	Return a list of cells, each cell containing three 2Xn_points sized array
	"""
	l1 = text.strip().split('CELL_START')
	l2 = [r.strip().split('\n') for r in l1]
	l3 = [r for r in l2 if 'SPOTS_START' in r]
	l4 = [[r.strip().split('\t') for r in l] for l in l3]
	l5 = [[r for i,r in enumerate(l) if '_POS' in r[0] or i > l.index(['SPOTS_START']) + 1] for l in l4]
	l6 = [[
		[r[3][1:], r[4][1:]], 
		[r[0][1:],r[1][1:]], 
		(numpy.reshape(list(itertools.chain(*[e[slice(19, 25, 2)] for e in r[6:]])), (-1, 3)).astype(int) 
		+ numpy.reshape(list(itertools.chain(*[e[slice(20, 26, 2)] for e in r[6:]])), (-1, 3)).astype(int))/2 if len(r[6]) > 4 else numpy.array(r[6:-1]).astype(float)/103]
		 for r in l5]
	l7 = [[numpy.reshape(r[0], (2, -1)).astype(int), numpy.reshape(r[1], (2, -1)).astype(int), numpy.reshape(r[2], (-1, 3)).T[[1, 0, 2]].astype(int)] for r in l6]
	return l7


def _stack_indices(r):
	prune_3d = lambda m: m[[0, 1, 3]] if m.shape[0] > 3 else m
	try:
		l = [prune_3d(numpy.vstack([e, [[i, ] * len(e[0])]])).astype(numpy.int16) for i, e in enumerate(r)]
		return numpy.hstack(l)
	except ValueError:
		warnings.warn('Unconsistent simulation found, ignoring.')
		return numpy.nan


def gather_paths(root):
	paths = []
	for path, subdirs, files in os.walk(root):
		for name in files:
			if name.lower().endswith('txt'):
				paths.append(os.path.join(path, name))

	return paths



def sort_and_arrange_paths(l):
	return [(p, p.split('/')[-2]) for p in sorted(l)]



def format(p, t):
	with open(p) as f:
		text = f.read()

	l = parse_text(text)

	return [[_stack_indices(e), t] for e in l]


def show(m):
	indices, gene = m
	ind = indices - numpy.min(indices, axis=1).reshape(-1, 1)

	m = numpy.zeros(numpy.max(ind, axis=1) + 1)

	m[tuple(ind)] = 1
	m *= numpy.array([1,2,3]).reshape((1,1,3))

	f, ax = plt.subplots(figsize=(10, 10))
	ax.imshow(numpy.max(m, axis=2))
	ax.set_title(gene)
	f.show()


def main():

	paths = gather_paths(DATA_PATH)

	tups = sort_and_arrange_paths(paths)

	biglist = [format(p, t) for p,t in tqdm.tqdm(tups)]

	bigarray = numpy.vstack(biglist)

	df = pandas.DataFrame(bigarray, columns = ['pos', 'gene'])
	le = LabelEncoder()
	df['labels'] = le.fit_transform(df.gene)

	df.pos = [m-numpy.min(m, axis=1).reshape(-1, 1) for m in tqdm.tqdm(df.pos)]

	return df



