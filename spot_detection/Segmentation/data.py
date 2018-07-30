from FishAnalyser import *
from multiprocessing import Pool

DATA_PATH = '/Users/remydubois/Desktop/Remy/_REMY/Opera_Conf/3D/'			


def gather_paths(root):
	paths = []
	for path, subdirs, files in os.walk(root):
		for name in files:
			if name.lower().endswith('tif') and not 'ch1' in name:
				paths.append(os.path.join(path, name))

	return paths


def sort_and_arrange_paths(l):
	dapi_paths = [p for p in l if 'ch2' in p]
	cm_paths = [p for p in l if 'ch3' in p]
	dapi_paths.sort()
	cm_paths.sort()

	return pandas.DataFrame({'dapi_paths': dapi_paths, 'cm_paths': cm_paths})


def segment(path):
	im = DAPIimage()
	im.load(path, shape=None, dtype='uint16')
	im.segment(sg=FastNuclei())
	return im.nucleis 


def get_mip(path):
	im = DAPIimage()
	im.load(path, shape=None, dtype='uint16')
	return numpy.amax(im.image_raw, axis=2)


def process_one_image(dapi, cm):
	gt = segment(dapi)
	x = get_mip(cm)

	return x, gt


def main():
	paths = gather_paths(DATA_PATH)
	df_paths = sort_and_arrange_paths(paths)

	pool = Pool(processes=4)
	pool.map(process_one_image, paths)





