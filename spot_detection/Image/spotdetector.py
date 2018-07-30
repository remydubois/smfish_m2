from skimage import measure, feature
from .utils import get_detection_params, gmm_threshold, extract_cube
import numpy
from scipy.ndimage import gaussian_filter, gaussian_laplace
import math
from math import sqrt, log
from Image.filters import FFT
from skimage.util import img_as_float
from skimage.feature.peak import peak_local_max
import time
from skimage.feature.blob import _prune_blobs
from joblib import Parallel, delayed


class SpotDetector(object):

    def __init__(self, **params):
        self.params = {}
        self.method = None
        # p = get_detection_params()
        # p.update(params)
        self.params.update(**params)

    def locate(self, im):
        return self.method(im, **self.params)


class LocalMax(SpotDetector):
    """
    LocalMax detector from skimage. It should not return images in this weird shape like it is doing now:
    returns tuples (y, x, z) instead of (x, y, z).

    """

    def __init__(self, **params):
        super(LocalMax, self).__init__(**params)
        method = feature.peak_local_max

        self.method = method  # lambda *args, **kwargs: numpy.array(method(*args, **kwargs))[:, [0, 1, 2]]


class DoG(SpotDetector):

    def __init__(self, **kwargs):
        super(DoG, self).__init__(**kwargs)

        def method(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold='GMM', filter=False):

            mip = numpy.amax(image, 2)

            # GMM threshold on the MIP is enough to get rid of the noise
            if threshold == 'GMM':
                threshold = gmm_threshold(mip, n_components=3)

            peaks = peak_local_max(image, threshold_abs=threshold, footprint=numpy.ones((8, 8, 8)))

            if peaks.size == 0:
                return numpy.empty((0, 3))

            # k such that min_sigma*(sigma_ratio**k) > max_sigma
            k = int(log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

            # a geometric progression of standard deviations for gaussian kernels
            sigma_list = numpy.array([min_sigma * (sigma_ratio ** i)
                                      for i in range(k + 1)])

            if filter:
                image = FFT().apply(image)

            gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

            # computing difference between two successive Gaussian blurred images
            # multiplying with standard deviation provides scale invariance
            dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                          * sigma_list[i] for i in range(k)]

            image_cube = numpy.stack(dog_images, axis=-1)

            inds_shapes = map(lambda p: numpy.argmax(image_cube[tuple(p)]), peaks)

            shapes = [sigma_list[i] for i in inds_shapes]

            blobs = numpy.hstack((peaks.astype(numpy.float64), numpy.array(shapes).reshape(-1, 1)))

            return blobs

        self.method = method


class LoG(SpotDetector):

    def __init__(self, **kwargs):
        super(DoG, self).__init__(**kwargs)

        def method(im, **kwargs):
            spots = feature.blob_log(im, **kwargs)
            return spots

        self.method = method
