from FishAnalyser import *
from multiprocessing import Pool
from skimage.segmentation import find_boundaries

DATA_PATH = '/Users/remydubois/Desktop/Remy/_REMY/Opera_Conf/3D/'
TARGET_PATH = '/Users/remydubois/Desktop/segmented/'

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

def enhance_inner_bounds(lab):
    
    b = find_boundaries(lab, mode='outer', connectivity=8)
    b[lab==0] = 0
    b = dilation(b, disk(3))
    lab2 = (lab > 0).astype(int)
    lab2[b>0] = 2
    
    return lab2

def segment_nuc(path):
    im = DAPIimage(verbose=0)
    im.load(path, shape=None, dtype='uint16')
    im.image_raw = im.image_raw[..., 15:25]
    im.segment(sg=FastNuclei())
    out = im.nucleis
    return resize(out, (512, 512), preserve_range=True, mode='constant')



def segment_nuc_and_cyt(dapi, cm):
    nuc = DAPIimage(verbose=0)
    nuc.load(dapi, shape=None, dtype='uint16')
    nuc.segment(sg=FastNuclei())
    cyt = CYTimage(nuc.nucleis, verbose=0)
    cyt.load(cm, shape=None, dtype='uint16')
    cyt.segment(sg=CytoSegmenter2())
    out = cyt.cells
    return numpy.amax(cyt.image_raw, 2), out, nuc.nucleis


def get_mip(path):
    im = DAPIimage(verbose=0)
    im.load(path, shape=None, dtype='uint16')
    out = numpy.amax(im.image_raw, axis=2)
    return resize(out, (512, 512), preserve_range=True, mode='constant')


def process_one_image(paths):
    dapi, cm = paths
    # gt = segment_nuc(dapi).astype(numpy.uint16)
    mip, cm, nuc = segment_nuc_and_cyt(dapi, cm)
    mip = resize(mip, (512, 512), preserve_range=True, mode='constant')
    cm = numpy.ceil(resize(enhance_inner_bounds(cm), (512, 512), preserve_range=True, mode='constant')).astype(numpy.uint8)
    nuc = numpy.ceil(resize((nuc > 0).astype(int), (512, 512), preserve_range=True, mode='constant')).astype(numpy.uint8)


    return numpy.stack([mip, cm, nuc], axis=0)


def main():
    paths = gather_paths(DATA_PATH)
    df_paths = sort_and_arrange_paths(paths)

    examples = [process_one_image(r) for r in tqdm.tqdm(df_paths.values[:])]

    out = numpy.stack(examples, axis=0)

    numpy.save(TARGET_PATH + 'cm_IB_dapi_conf.npy', out)


if __name__ == '__main__':
    main()
