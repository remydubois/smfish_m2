# Here define functions to use on batches of images. They have to be decorated with adapt.
from .tools import adapt
from .image import *


# @adapt
def compute_snr(im):
    im.detect_and_fit(threshold='GMM')
    im.compute_snr()
    im_name = im.name.split('/')[-1].lower().replace('.tif', '')
    microscope = im.name.split('/')[6]
    gene = im.name.split('/')[8]

    return im_name, microscope, gene, len(im.spots), im.SNR


@adapt
def segment_cells(im):
    im.segment()

    mask = im.cells

    return numpy.unique(mask).shape[0] - 1


@adapt
def detect_spots(im, threshold=0.):
    heights = im.detect_and_fit(threshold=threshold, return_profile=True)
    return im.name, heights


# @adapt
def detect_and_seg(t):
    """
    There has to be a try-catch because some wells have a fish image but no dapi or vice-versa.

    :param p1:
    :param p2:
    :return:
    """

    # fish, mask = t
    fish = FQimage(verbose=0)
    mask = DAPIimage(verbose=0)
    fish.load(t[0])
    mask.load(t[1])
    im_name = fish.name.split('/')[-1].lower().replace('.tif', '')
    microscope = fish.name.split('/')[6]
    gene = fish.name.split('/')[8]

    mask.segment()
    # mask.show_cells()
    cells = numpy.unique(mask.nucleis).shape[0] - 1  # dont forget background

    fish.detect_and_fit(threshold='GMM')
    fish.compute_snr(boxplot=False)
    snr = numpy.mean(fish.SNR)
    spots = len(fish.spots)

    # print('.', end="", flush=True)

    # garbage
    del mask
    del fish

    return im_name, microscope, gene, cells, spots, snr


def seg_both(t):
    dapi = DAPIimage(verbose=0)

    dapi.load(t[0])

    dapi.segment(FastNuclei())
    nucleis = dapi.nucleis

    mask = CYTimage(nucleis, verbose=0)
    mask.load(t[1])
    mask.segment(CytoSegmenter2(clear_borders=True))

    cells = mask.cells

    pile = numpy.stack([cells, nucleis])
    out = []
    for r in regionprops(cells):
        minr, minc, maxr, maxc = r.bbox
        extract = pile[:, minr-1:maxr+1, minc-1:maxc+1].copy()
        extract[:, extract[0] != r.label] = 0
        # plt.imshow(extract[0])
        sequences = []
        flag = False
        for ch in extract:
            boundaries = find_boundaries(ch, mode='inner')
            X = numpy.vstack(numpy.nonzero(boundaries)).T
            Y = orient_set(X)
            if Y is None:
                flag = True
                break
            sequences.append(Y)

        if flag:
            continue
        out.append(sequences)

    return zip(*out)