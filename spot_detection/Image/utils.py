import numpy
import functools
import math
from scipy import spatial
import warnings
import itertools
from skimage.feature.blob import _compute_disk_overlap, _compute_sphere_overlap
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from skimage.segmentation import find_boundaries


def get_image_path():
    # return '/Users/remydubois/Dropbox/Remy/Python/CodeFragments/TestData/Experiment-453.czi #001 - C=1-1.tif'
    return '/Users/remydubois/Dropbox/Remy/Python/CodeFragments/TestData/171208_00_w1TIRF-561.TIF'


def get_detection_params():
    # 'threshold_abs': 5000,
    # return {'min_distance': 10}
    return {}


def get_filtering_params():
    return {'sigma_bgd': 5, 'sigma_spots': 0.5}


def get_fitting_params():
    return


def extend_class(f):
    """
    Decorator used to add methods defined in another module.

    :param f: the function to add to the cls object.
    :return: the cls object from the wrapper below.
    """

    def wrapper(cls):
        setattr(cls, f.__name__, f)

        return cls

    return wrapper


def Gaussian1D(center_x, background, height, width_x):
    return lambda x: background + height * numpy.exp(-(((center_x - x) / (math.sqrt(2) * width_x)) ** 2) * 2)


def Gaussian3D(center_x, center_y, center_z, height, width_x, width_y, width_z):
    # print('locals Gauss', locals())
    return lambda x, y, z: height * numpy.exp(-(((center_x - x) / (math.sqrt(2) * width_x)) ** 2 +
                                                ((center_y - y) / (math.sqrt(2) * width_y)) ** 2 +
                                                ((center_z - z) / (math.sqrt(2) * width_z)) ** 2) * 2)


def MixtureGaussian3D(centers, backgrounds, *params):
    params = numpy.array(params).reshape(-1, 4)
    # z = zip(centers, backgrounds, params)
    # n = next(z)
    # # print('n', n)
    # print(list(itertools.chain(*n)))
    # print('.', end='', flush=True)
    return lambda x, y, z: sum(
        [Gaussian3D(*itertools.chain(*p))(x, y, z) for p in zip(centers, backgrounds, params)])


def extract_cube(point, side=7):
    """
    Slices the images in order to extract a cube of side length 'side' center on 'point'.
    It actually allows extraction of rectangles. In this case, the side argument precises the length of the side of each
    dimension (x, y , z).

    :param point: poi
    :param side: side length (must be odd)
    :return: slices
    """
    if not hasattr(side, '__iter__'):
        side = [side, side, side]

    down = [int(s // 2) for s in side]
    point = [int(p) for p in point]
    out = [slice(max(0, point[0] - down[0]), max(0, point[0] + down[0] + 1)),
           slice(max(0, point[1] - down[1]), max(0, point[1] + down[1] + 1)),
           slice(max(0, point[2] - down[2]), max(0, point[2] + down[2] + 1))
           ]

    return out


def extract_sphere(point, radius=5):
    xs = slice(point[0] - radius, point[0] + radius)


def trackcalls(func):
    """
    This decorator will further be used in order to know whether a function was called or not on a given object.
    Use example: know whether an image has already been filtered or not.
    An alternative approach would be to store booleans like 'filtered', 'spot_detected' as image instance attributes.

    :param func: the function to be tracked against its calls.
    :return: a wrapper (lookalike function) with the has_been_called attribute.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)

    wrapper.has_been_called = False
    return wrapper


def link(cls, func):
    """
    This function is used, when any of 'filter', 'detect', 'fit', 'compute_snr' is called,
    to make sure that they are called in the right order (i.e filter before detect spots for instance).
    In case one uses detect_spots before filter, filter will automatically be called with its default
    parameters.
    Use:
    When 'func' is called, then the previous function in the chain of order will be checked to know whether it has
    already been applied or not.

    :param cls, func: the class where to seek for preceding functions to func in the chain of order.
    :return: the wrapped func.
    """

    @trackcalls
    def f(*args, **kwargs):
        chain = ['load', 'filter', 'detect_spots', 'fit_spots', 'compute_snr']

        previous_step = chain[max(0, chain.index(func.__name__) - 1)]
        # Do not check whether the image was loaded or not. Cant decide for the user...
        if previous_step == 'load':
            return func(*args, **kwargs)
        # else ...
        previous_func = getattr(cls, previous_step)
        if not previous_func.has_been_called:
            if args[0]._verbose > 0:
                print("'%s' has been called but '%s' has not been called. Calling with default arguments." % (
                    func.__name__.title(), previous_step.title()))
            previous_func(*args, **kwargs)

        return func(*args, **kwargs)

    return f


def chain_methods(cls):
    """
    This one is finally a decorator for the whole image class which wraps each of its methods (except init and show)
    into the link decorator described above.

    :param cls: the class one wishes to wrap the methods.
    :return: the same object, once setattr was applied.
    """
    method_list = [func for func in dir(cls) if callable(getattr(cls, func))
                   and not func.startswith("__")
                   and not func == 'load'
                   and not func.startswith("show")
                   and not func == 'segment'
                   and not func == 'split'
                   ]

    for m in method_list:
        setattr(cls, m, link(cls, getattr(cls, m)))

    return cls


def get_focus_size():
    return 20


def find_nearest_region(im, x, y):
    tmp = im[x, y]
    im[x, y] = 0
    r, c = numpy.nonzero(im)
    im[x, y] = tmp
    min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
    return im[r[min_idx], c[min_idx]]


def ellipse_in_shape(radii, shape, center, hole_radius_ratio=0):
    """Generate coordinates of points within ellipse bounded by shape.
    Parameters
    ----------
    shape :  iterable of ints
        Shape of the input image.  Must be length 3.
    center : iterable of floats
        (row, column) position of center inside the given shape.
    radii : iterable of floats
        Size of two half axes (for row and column)
    rotation : float, optional
        Rotation of the ellipse defined by the above, in radians
        in range (-PI, PI), in contra clockwise direction,
        with respect to the column-axis.
    Returns
    -------
    rows : iterable of ints
        Row coordinates representing values within the ellipse.
    cols : iterable of ints
        Corresponding column coordinates representing values within the ellipse.
    """
    r_lim, c_lim, d_lim = numpy.ogrid[0:float(shape[0]), 0:float(shape[1]), 0:float(shape[2])]
    r_org, c_org, d_org = center
    r_rad, c_rad, d_rad = radii
    r, c, d = (r_lim - r_org), (c_lim - c_org), (d_lim - d_org)
    distances = (r / r_rad) ** 2 \
                + (c / c_rad) ** 2 \
                + (d / d_rad) ** 2

    if hole_radius_ratio > 0:
        # return numpy.multiply(numpy.nonzero(distances < 1), numpy.nonzero(distances > hole_radius_ratio))
        return numpy.nonzero(numpy.logical_and(distances > hole_radius_ratio ** 2, distances <= 1))
    else:
        return numpy.nonzero(distances <= 1)


def ellipsis(radii, shape=None, center=None, hole_radius_ratio=0.):
    if len(radii) < 3:
        raise ValueError('Please use skimage ellipse for 2d ellipse. Otherwise provide three radii.')

    if shape is None:
        shape = tuple(map(lambda x: x * 2 + 1, radii))
        center = radii

    r_lim, c_lim, d_lim = numpy.ogrid[0:float(shape[0]), 0:float(shape[1]), 0:float(shape[2])]
    r_org, c_org, d_org = center
    r_rad, c_rad, d_rad = radii
    r, c, d = (r_lim - r_org), (c_lim - c_org), (d_lim - d_org)
    distances = ((r / r_rad) ** 2 \
                 + (c / c_rad) ** 2 \
                 + (d / d_rad) ** 2)

    if hole_radius_ratio > 0.:
        return numpy.logical_and(distances > hole_radius_ratio ** 2, distances <= 1), \
               distances <= hole_radius_ratio ** 2
    else:
        return distances <= 1


def blob_overlap(blob1, blob2):
    """Finds the overlapping area fraction between two blobs.
    Returns a float representing fraction of overlapped area.
    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).
    """
    n_dim = len(blob1) - 1

    # extent of the blob is given by sqrt(2)*scale
    r1 = blob1[-1] * 2
    r2 = blob2[-1] * 2

    d = math.sqrt(numpy.sum((blob1[:-1] - blob2[:-1]) ** 2))
    if d > r1 + r2:
        return 0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        return 1

    if n_dim == 2:
        return _compute_disk_overlap(d, r1, r2)

    else:  # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        return _compute_sphere_overlap(d, r1, r2)


def prune_blobs(blobs_array, overlap):
    """Eliminated blobs with area overlap.

    Actually eliminates blobs with overlapping vicinity (vicinity being defined as 4*sigma) in order to identify noise
    around a blob. As the overlapping volume of two ellipsis is not straghtforward to compute, blobs are assumed to be
    sphere even if it is not the case in FISH images. However, considering them as sphere leads to overprudence: more
    blobs will be deleted than what is actually needed but the impact should be rather small.

    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 (or 4) values,
        ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
        where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
        and ``sigma`` is the standard deviation of the Gaussian kernel which
        detected the blob.
        This array must not have a dimension of size 0.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """
    sigma = blobs_array[:, -1].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - 1)
    tree = spatial.cKDTree(blobs_array[:, :-1])
    pairs = numpy.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            # Vicinity is here only 2*sigma because blob_overlap already takes a * 2 factor into account.
            blob1_vicinity = blob1.copy()
            blob1_vicinity.itemset(3, 2 * blob1[3])
            blob2_vicinity = blob2.copy()
            blob2_vicinity.itemset(3, 2 * blob2[3])
            if blob_overlap(blob1_vicinity, blob2) > overlap:
                blob1[-1] = 0
            if blob_overlap(blob2_vicinity, blob1) > overlap:
                blob2[-1] = 0

    return numpy.array([b for b in blobs_array if b[-1] > 0])


def adjust_and_filter(blobs_array, image):
    spots = blobs_array[:, :3].copy().astype(int)
    intensities = [image[tuple(s)] for s in spots]
    bins = [i * numpy.iinfo(image.dtype).max / 255 for i in range(0, 255)]
    counts, _ = numpy.histogram(intensities, bins)
    profile = numpy.cumsum(counts[::-1])[::-1]
    hess = numpy.gradient(numpy.gradient(profile))

    ind = numpy.argmax(hess)

    f = filter(lambda t: t[0] > bins[ind + 1], zip(intensities, blobs_array))

    return numpy.stack(list(zip(*f))[1])


def gmm_threshold(img, n_components=3, plot=False, border=-1, return_aic=False):
    if img.ndim > 2:
        warnings.warn('Working on a 3d image, expect high computational time.')
    flag = False
    GMM = GaussianMixture(n_components=n_components)
    GMM.fit(img.reshape(-1, 1))
    # Find threshold
    xs = numpy.linspace(img.min(), img.max(), 1000)
    ys = GMM.predict(xs.reshape(-1, 1))
    inds = numpy.nonzero(ys[1:] - ys[:-1])[0]
    # print(ys)
    thresh = xs[inds[border]]
    # otsu = threshold_otsu(img)
    # means = numpy.sort(GMM.means_.ravel())
    # thresh = numpy.mean([means[1:], means[:-1]], axis=0)[border]
    # print(numpy.mean([means[1:], means[:-1]], axis=0))
    if plot:
        f, ax = plt.subplots(figsize=(8, 8))
        ax.hist(img.ravel(), bins=255, label='Observed data')
        sample, _ = GMM.sample(numpy.prod(img.shape))
        ax.hist(sample[sample > img.min()], bins=255, alpha=0.5, label='Model')
        ax.axvline(thresh, c='r')
        # ax.axvline(275, c='g', linestyle='dashed')
        # ax.axvline(otsu, c='b')
        ax.text(thresh-75, 10,  'GMM Threshold', rotation=90, color='r')
        # ax.text(275 + 25, 1.e5, '"optimal" Threshold', rotation=90, color='g')
        # ax.text(otsu + 25, 10, 'Otsu Threshold', rotation=90, color='b')
        # ax.set_xticks(list(range(0, 3000, 500)) + [thresh])
        ax.set_yscale('log')
        ax.set_title('Cell mask pixel intensity histogram (blue) and fitted GMM (light orange)')
        ax.set_xlabel('Pixel intensity values')
        ax.set_ylabel('Pixels count (log)')
        ax.legend()
        # ax.set_xlim([0, 3000])
        f.show()
    if return_aic:
        return thresh, GMM.aic(img.reshape(-1, 1))
    return thresh

def KMeans_threshold(img, n_components=3, plot=False, border=-1, return_score=False):
    if img.ndim > 2:
        warnings.warn('Working on a 3d image, expect high computational time.')

    flag = False

    KM = KMeans(n_clusters=n_components)
    KM.fit(img.reshape(-1, 1))

    # Find threshold
    xs = numpy.linspace(img.min(), img.max(), 1000)
    ys = KM.predict(xs.reshape(-1, 1))
    inds = numpy.nonzero(ys[1:] - ys[:-1])[0]
    # print(ys)

    if plot:
        f, ax = plt.subplots(figsize=(8, 8))
        ax.hist(img.ravel(), bins=255)
        if border is not None:
            ax.axvline(xs[inds[border]], c='r')
        else:
            for i in inds:
                ax.axvline(xs[i], c='r')
        ax.set_yscale('log')
        ax.set_title(', '.join(map(str, numpy.sort(KM.cluster_centers_.ravel()))))
        f.show()

    if return_score:
        return xs[inds[border]], KM.score(img.reshape(-1, 1))

    if border is None:
        return [xs[i] for i in inds]

    return xs[inds[border]]


def multiclass_otsu(img, n_components=2, plot=False, border=-1, return_score=False):
    if img.ndim > 2:
        warnings.warn('Working on a 3d image, expect high computational time.')

    flag = False

    KM = KMeans(n_clusters=n_components)
    KM.fit(img.reshape(-1, 1))

    # Find threshold
    xs = numpy.linspace(img.min(), img.max(), 1000)
    ys = KM.predict(xs.reshape(-1, 1))
    inds = numpy.nonzero(ys[1:] - ys[:-1])[0]
    # print(ys)

    if plot:
        f, ax = plt.subplots(figsize=(8, 8))
        ax.hist(img.ravel(), bins=255)
        if border is not None:
            ax.axvline(xs[inds[border]], c='r')
        else:
            for i in inds:
                ax.axvline(xs[i], c='r')
        ax.set_yscale('log')
        ax.set_title(', '.join(map(str, numpy.sort(KM.cluster_centers_.ravel()))))
        f.show()

    if return_score:
        return xs[inds[border]], KM.inertia_

    if border is None:
        return [xs[i] for i in inds]

    return xs[inds[border]]


def orient_set(X):
    """
    From an unordered set of points, return a trajectory (in the physic sense of the term) which is by definition ordered.
    In practice, this is used to obtain a directed line from a cell boundary matrix (i.e. the matrix of points defining
    the border of the cell).

    In terms of speed, neighbors are queried for each point of the border using sklearn's KDTree. The whole routine
    runs in less than 100ms for a set of ~ 1000 points.

    This method assumes of course that border is cyclic.

    :param m: the matrix of points of shape (n_points, n_dimensions)
    :return: the same matrix, ordered
    """

    tree = KDTree(X, leaf_size=2)
    # print(X)
    path = []

    start = 0
    path.append(start)
    dist, ind = tree.query([X[path[-1]]], k=3)

    # Impose first step for the direction
    prochain = ind[0][1]

    step = 0
    while True:
        path.append(prochain)

        # Get 8 neighbors for the whole connectivity of each pixel:
        # one has to look at every possible neighbor for each pixel.
        _, ind = tree.query([X[path[-1]]], k=8, return_distance=True, sort_results=True)
        for p in ind[0][1:]:
            if p not in path[-30:]:
                prochain = p
                break
        step += 1
        if prochain in path[:30]:
            break

        # Escapes in case the algo is looping indefinitely
        if step > X.shape[0]:
            return None

    out = numpy.stack([X[i] for i in path])

    return out

def mask_boundaries(seg):
    bounds = find_boundaries(seg)
    return numpy.ma.masked_where(bounds==0, 0)