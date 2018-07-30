import matplotlib.pyplot as plt
from matplotlib import colors
from .utils import *
import tqdm
from itertools import product
import numpy
import numpy
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from matplotlib.patches import Rectangle
from skimage.transform import resize


def show_image(self, show=True):
    f, ax = plt.subplots(figsize=(18, 10))
    b = ax.imshow(numpy.amax(self.image_raw, axis=2), cmap='inferno')
    ax.set_title('raw image')
    f.colorbar(b)
    if show:
        f.show()
        return None
    else:
        return f, ax


def show_spots(self):
    """
    This method draws circle around spots on the images. T-uple of the spots are not read in the original order
    BECAUSE axis 0 for numpy is row which is the vertical axis, so the y (2nd dimension) for matplotlib.
    :param self:
    :return:
    """
    f, axs = self.show_image(False)
    for p in self.spots:
        axs.add_patch(
            plt.Circle((p.coordinates[1], p.coordinates[0]), 2 * p.model.params['width_x'], fill=False, color='r'))
    f.show()


def show_detection_profile(self):
    try:
        assert self.spots is not None
        intensities = [self.image_raw[t.coordinates] for t in self.spots]
        intensities.sort()
        counts = [sum(1 for _ in filter(lambda x: x > f, intensities)) for f in range(int(numpy.max(intensities)))]
        f, ax = plt.subplots()
        ax.plot(counts[::-1])
        f.show(0)

    except AssertionError:
        print('Detect spots first.')


def show_model(self):
    modeled = numpy.fromfunction(self.mixture_model, shape=self.image_raw.shape)
    # print(modeled)
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    ax1.imshow(numpy.amax(modeled, axis=2))
    ax1.set_title('Modeled image.')
    ax2.imshow(numpy.amax(self.image_raw, axis=2))
    ax2.set_title('Raw image.')
    plt.show()


def show_nucleis(self, show=True):
    f, axes = plt.subplots(figsize=(8, 8))

    if self.image_raw.shape[0] > 1024:
        im = resize(numpy.amax(self.image_raw, 2), (512, 512), mode='constant', preserve_range=True).astype(
            self.image_raw.dtype)
        nuc = resize(self.nucleis, (512, 512), mode='constant', preserve_range=True).astype(
            self.nucleis.dtype)
    else:
        im = numpy.amax(self.image_raw, 2)
        nuc = self.nucleis
    axes.imshow(im)
    boundaries = find_boundaries(nuc, mode='inner')

    masked_bound = numpy.ma.masked_where(boundaries == 0, boundaries)

    cmap = colors.ListedColormap(['red'])
    axes.imshow(masked_bound, cmap=cmap)
    if show:
        f.show()
    else:
        return f


def show_cells(self, show=True):

    if self.cells.ndim == 2:
        f, ax = plt.subplots(figsize=(8, 8))

        im = numpy.amax(self.image_raw, 2)

        ax.imshow(im)
        boundaries_nuc = find_boundaries(self.nuclei_image)
        boundaries_cyt = find_boundaries(self.cells)
        ax.imshow(numpy.ma.masked_where(boundaries_nuc == 0, boundaries_nuc), cmap=colors.ListedColormap(['red']))
        ax.imshow(numpy.ma.masked_where(boundaries_cyt == 0, boundaries_cyt), cmap=colors.ListedColormap(['blue']))

        f.show()

    if self.cells.ndim == 3:
        f = plt.figure(figsize=(10, 10))
        ax = f.gca(projection='3d')

        X, Y = numpy.meshgrid(numpy.arange(self.cells.shape[0]), numpy.arange(self.cells.shape[0]))
        Z = self.cells
        surf = ax.plot_surface(X, Y, numpy.sum(Z > 0, axis=-1), cmap='viridis',
                               linewidth=0, antialiased=False)
        ax.set_zlim(0, 60)

        f.show()

    return f