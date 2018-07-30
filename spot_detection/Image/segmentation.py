import numpy
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk, reconstruction, remove_small_holes, remove_small_objects, opening
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border, find_boundaries
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage.morphology import closing
from collections import Counter
from scipy.sparse import coo_matrix
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.filters import gaussian, laplace

from .utils import *
import time
from skimage.morphology import dilation
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import FunctionTransformer
from skimage.morphology import binary_opening
import networkx as nx
import tqdm


class Segmenter(object):

    def __init__(self):
        self.method = None

    def segment(self, im, **kwargs):
        # if not type(im) == FQimage:
        #     raise TypeError('Input must be an FQimage')
        out = self.method(im, **kwargs)

        # Store in a Sparse matrix for economy.
        im.cells = coo_matrix(out)


class NucleiSegmenter(Segmenter):

    def __init__(self, min_size=200):
        """
        This method is intended at segmenting the cells in FQImage on Mask images (not FISH). However basic, it seems
        to give a rather good
        approximation. The workflow is MIP -> local grad -> Otsu thresholding -> Connected components labelling ->
        Filtering components based on their size (using either hand-threshold or KMeans to distinguish actual cells
        from noise components.

        Those computation relie on the assumption that the strongest gradient of a cell is at its border. Actually,
        tit might happen that some nuclei show very high gradient, therefore, I have to check that elements are not
        imbricated.
        :param min_size: Either 'auto' (for KMeans call) or an int. Minimum size of a component to be considered
        a cell.
        :return: None.
        :param min_size:

        TODO investigate skimage 'area' parameter of regionprops instances (maybe shorter code).
        """

        super(NucleiSegmenter, self).__init__()

        def segment(img):
            im = img.copy()
            # if img.shape[0] > 1024:
            #     im = resize(img, (512, 512, img.shape[-1]), mode='constant', preserve_range=True).astype(img.dtype)
            # else:
            #     im = img.copy()
            slices = [im.shape[-1] // 6, 2 * im.shape[-1] // 6, 3 * im.shape[-1] // 6, 4 * im.shape[-1] // 6,
                      5 * im.shape[-1] // 6]

            grads = numpy.stack([rank.gradient(im[:, :, z], disk(2)) for z in slices])
            gradients = numpy.amax(grads, 0)
            thresh = threshold_otsu(gradients)
            binary = (gradients > thresh).astype(int)
            labels_1 = opening(label(remove_small_holes(binary, 10000)), disk(2))

            binary = dilation(binary, disk(10))

            new_thresh = rank.otsu(gradients, disk(20), mask=binary)
            new_binary = (gradients > new_thresh).astype(int)
            new_binary[binary == 0] = 0

            new_binary = remove_small_holes(new_binary.astype(bool), 10000)
            labels_2 = label(new_binary)
            new_labels = (labels_1 + labels_2) > 0
            new_labels = label(remove_small_holes(new_labels, 100))

            new_labels = remove_small_objects(new_labels, 100)

            if numpy.unique(new_labels).shape[0] == 1:
                raise UserWarning('No nucleis detected.')

            return new_labels

        self.method = segment


class NucleiSegmenter2(Segmenter):

    def __init__(self, min_size=100):
        """
        This method is intended at segmenting the cells in FQImage on Mask images (not FISH). However basic, it seems
        to give a rather good
        approximation. The workflow is MIP -> local grad -> Otsu thresholding -> Connected components labelling ->
        Filtering components based on their size (using either hand-threshold or KMeans to distinguish actual cells
        from noise components.

        Those computation relie on the assumption that the strongest gradient of a cell is at its border. Actually,
        tit might happen that some nuclei show very high gradient, therefore, I have to check that elements are not
        imbricated.
        :param min_size: Either 'auto' (for KMeans call) or an int. Minimum size of a component to be considered
        a cell.
        :return: None.
        :param min_size:

        TODO investigate skimage 'area' parameter of regionprops instances (maybe shorter code).
        """

        super(NucleiSegmenter2, self).__init__()

        def segment(img):
            im = img.copy()
            # if img.shape[0] > 1024:
            #     im = resize(img, (512, 512, img.shape[-1]), mode='constant', preserve_range=True).astype(img.dtype)
            # else:
            #     im = img.copy()
            mip = numpy.amax(im, 2)
            transformed = numpy.log1p(mip)
            GMM = GaussianMixture(n_components=3)

            GMM.fit(transformed.reshape(-1, 1))
            nucleis_label = numpy.argmax(GMM.means_.ravel())
            xs = numpy.log1p(numpy.arange(mip.min(), mip.max()).reshape(-1, 1))
            ys = GMM.predict(xs.reshape(-1, 1))
            thresh = xs[numpy.argmax(ys == nucleis_label)]
            nucs = transformed > thresh
            nucs = remove_small_holes(nucs, 20).astype(int)
            labels = label(nucs)
            labels = remove_small_objects(labels, min_size).astype(int)

            if numpy.unique(labels).shape[0] == 1:
                raise UserWarning('No nucleis detected.')

            return labels

        self.method = segment


class FastNuclei(Segmenter):

    def __init__(self, min_size=100):
        super(FastNuclei, self).__init__()

        def segment(img):
            im = img.copy()
            mip = numpy.amax(im, 2)
            binary = mip > threshold_otsu(mip)
            binary = remove_small_holes(binary, 20000)
            labels = label(binary)
            labels = remove_small_objects(labels, min_size)

            if numpy.unique(labels).shape[0] == 1:
                raise UserWarning('No nucleis detected.')

            return labels

        self.method = segment


class CytoSegmenter(Segmenter):

    def __init__(self):

        super(CytoSegmenter, self).__init__()

        def segment(cyto_image, nuclei_labels):

            if nuclei_labels is None:
                raise ValueError('A Nuclei label image must be provided.')

            if cyto_image.ndim < 3:
                raise ValueError('Cytoplasm image must be 3-dimensional.')

            im = cyto_image.copy()

            # if cyto_image.shape[0] > 1024:
            #     im = resize(cyto_image, (512, 512, cyto_image.shape[-1]), mode='constant')
            #     if cyto_image.dtype == numpy.uint8:
            #         im = img_as_ubyte(im)
            #     if cyto_image.dtype == numpy.uint16:
            #         im = img_as_uint(im)
            # else:
            #     im = cyto_image.copy()

            # Compute mask of cytoplasm to speed up process. Erase nuclei for better sensitivity of the otsu method.
            mask = numpy.amax(im, 2)
            mask[nuclei_labels > 0] = 0
            mask = (mask > threshold_otsu(mask)).astype(int)
            mask = remove_small_objects(mask, 20)
            mask = remove_small_holes(mask, 20)
            mask[nuclei_labels > 0] = 1

            # Build the seed for watershed segmentation by stacking up successive layers of non-null cytoplasm region.
            seed = numpy.sum(im, 2)
            seed = seed.max() - seed

            # Get the markers from the nuclei
            markers = numpy.zeros(seed.shape, dtype=int)
            for r in regionprops(nuclei_labels):
                markers[tuple(map(int, r.centroid))] = r.label

            # Start from scratch.
            target = numpy.zeros_like(nuclei_labels)

            target[nuclei_labels > 0] = -1
            seed[nuclei_labels > 0] = seed.min()
            cell_seg = watershed(seed, markers, mask=mask)

            return cell_seg

        self.method = segment


class CytoSegmenter2(Segmenter):

    def __init__(self, clear_borders=False):

        super(CytoSegmenter2, self).__init__()

        def segment(cyto_image, nuclei_labels, clear_borders=clear_borders):
            """
            TODO Issues with Opera WF whose MIP is terrible.

            :param cyto_image:
            :param nuclei_labels:
            :return:
            """

            if nuclei_labels is None:
                raise ValueError('A Nuclei label image must be provided.')

            if cyto_image.ndim < 3:
                raise ValueError('Cytoplasm image must be 3-dimensional.')

            im = cyto_image.copy()

            # Compute mask of cytoplasm to speed up process. Erase nuclei for better sensitivity of the otsu method.
            mip = numpy.amax(im, 2)
            # mip =
            base = numpy.log1p(mip)

            # tuples = [gmm_threshold(base, n_components=i, return_aic=True) for i in [2, 3]]
            binary = base > gmm_threshold(base, n_components=2)
            binary = remove_small_holes(binary, 15).astype(int)
            labels = label(binary)
            labels = remove_small_objects(labels, 50)
            mask = labels > 0

            # Build the seed for watershed segmentation by stacking up successive layers of non-null cytoplasm region.
            seed = numpy.sum(im, 2)
            seed = seed.max() - seed

            # Get the markers from the nuclei
            markers = numpy.zeros(seed.shape, dtype=int)
            for r in regionprops(nuclei_labels):
                markers[tuple(map(int, r.centroid))] = r.label

            seed[nuclei_labels > 0] = seed.min()
            mask[nuclei_labels > 0] = True
            cell_seg = watershed(seed, markers, mask=mask)

            if clear_borders:
                cell_seg = clear_border(cell_seg)

            return cell_seg

        self.method = segment


class CytoSegmenter3D(Segmenter):

    def __init__(self):

        super(CytoSegmenter3D, self).__init__()

        def segment(cyto_image, nuclei_labels):
            binaries = []

            thresholds = [threshold_otsu(numpy.log(cyto_image[:, :, i])) for i in range(35)]
            ceiling = max(thresholds)

            for i in range(35):
                slice = cyto_image[:, :, i]
                segmented = numpy.log1p(slice) > (thresholds[i] + ceiling) / 2
                segmented = binary_opening(segmented, disk(5))
                binaries.append(segmented)

            pile = numpy.zeros_like(cyto_image[:, :, 0])
            sizes = [b.sum() for b in binaries]
            start = 34
            # It is doable to prove that once the plateforme has been passed, the amount of pixels above otsu threshold
            # decreases. Therefore
            stop = numpy.argmax(sizes)

            stack = []
            for i in range(start, stop - 1, -1):
                p = cyto_image[:, :, i]
                pile += p
                seed = pile.copy()
                mask = binaries[i]
                seed[(nuclei_labels > 0) * mask] = seed.max()
                peaks = peak_local_max(pile * mask, labels=nuclei_labels * mask, num_peaks_per_label=1, indices=False)
                # Ensures that a given cell receives the same label throughout the different slices
                markers = peaks * nuclei_labels
                segmented = watershed(-seed, markers=markers, mask=mask)
                stack.append(segmented)

            return numpy.stack(stack, axis=-1)

        self.method = segment


class MyRag(nx.Graph):

    def __init__(self, labels, verbose=1, **kwargs):
        """
        Simpler implementation of RAGs than skimage's one. Roughly ~ 10x faster to build due to numpy tricks and single
        channel operations.
        Order of the operations performed is crucial. Inverting two of them might double build time.

        Whole segmentation process (building + pruning the graph + reconstructing labels) is done in ~ 1.2 secs versus
        ~ 13 secs on skimage.
        """

        super(MyRag, self).__init__(**kwargs)

        h = numpy.stack((labels[1:, :].ravel(), labels[:-1, :].ravel()), axis=1)
        v = numpy.stack((labels[:, 1:].ravel(), labels[:, :-1].ravel()), axis=1)
        edges = numpy.vstack((v, h))
        edges.sort(axis=1)
        edges = edges[edges[:, 0] != edges[:, 1]]
        edges = numpy.unique(edges, axis=0)
        for e in tqdm.tqdm(edges, disable=not bool(verbose)):
            self.add_edge(*e)

    def cut_threshold(self, threshold):
        to_remove = [(x, y) for x, y, d in self.edges(data=True) if d['distance'] >= threshold]

        self.remove_edges_from(to_remove)


class MeanColorRag(MyRag):

    def __init__(self, labels, raw):
        super(MeanColorRag, self).__init__(labels=labels)

        mean_per_label = ndi.measurements.mean(raw, labels, numpy.unique(labels))

        # nx.set_node_attributes(self, {i:ml for i, ml in enumerate(mean_per_label)}, 'mean_color')

        nx.set_edge_attributes(self,
                               {t: numpy.abs(mean_per_label[t[0]] - mean_per_label[t[1]]) for t in self.edges},
                               'distance')


class HistoColorRag(MyRag):
    def __init__(self, labels, raw):
        super(HistoColorRag, self).__init__(labels=labels, verbose=0)

        histogram, _, _ = numpy.histogram2d(labels.ravel(), raw.ravel(), bins=(numpy.unique(labels).shape[0], 1000))

        nx.set_node_attributes(self, {i: histo for i, histo in enumerate(histogram)}, 'histogram')

        def chisquaredist(histo1, histo2):
            return sum((a - b) ** 2 / (a + b + 1) for a, b in zip(histo1, histo2))

        nx.set_edge_attributes(self,
                               {t: chisquaredist(histogram[t[0]], histogram[t[1]]) for t in
                                tqdm.tqdm(self.edges, total=self.number_of_edges())},
                               'distance')


def cut_threshold(rag, labels, raw, threshold):
    g = rag.copy()
    to_remove = [(x, y) for x, y, d in g.edges(data=True) if d['distance'] >= threshold]
    g.remove_edges_from(to_remove)

    components = list(nx.connected_components(g))
    out = numpy.zeros_like(labels)

    # Allows to avoid filling all the background.
    zero = numpy.argmax([len(c) for c in components])

    for i, nodes in tqdm.tqdm(enumerate(components), total=len(components)):
        if i == zero:
            continue
        for n in nodes:
            out[labels == n] = i

    return out
