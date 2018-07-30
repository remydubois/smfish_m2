from scipy.ndimage.filters import convolve, gaussian_laplace, median_filter
import numpy
from skimage import filters


class Filter(object):

    def __init__(self):
        pass


class ConvFilter(Filter):

    def __init__(self, kernel=None, n_d=3, mode='constant'):

        super(ConvFilter, self).__init__()

        self._kernel = kernel

        self._n_d = n_d

        self._mode = mode

    # def convolve(self, input):
    #     return convolve(input=input, weights=self._kernel, mode=self._mode)

    def set_params(self, **params):
        for k in params.keys():
            setattr(self, k, params[k])

    @property
    def kernel(self):
        return self._kernel

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
        try:
            assert val in modes
            self._mode = val
        except AssertionError:
            print('Mode not in ' + ' ,'.join(modes))


class GaussianFilter(ConvFilter):

    def __init__(self, mu=0, sigma=1, size=(4, 4, 4)):
        super(GaussianFilter, self).__init__()

        if not hasattr(size, '__iter__'):
            size = tuple(size)

        self._kernel = numpy.random.normal(loc=mu, scale=sigma, size=size)
        self._size = size
        self._sigma = sigma
        self._mu = mu

    def convolve(self, input):
        return filters.gaussian(input, sigma=self._sigma)


class LoG(GaussianFilter):

    def __init__(self, **kwargs):
        super(LoG, self).__init__(**kwargs)

        self._kernel = 'Gaussian_Laplacian'
        self._mu = 0

    def convolve(self, input):
        return gaussian_laplace(input=input, sigma=self._sigma)


class FFT(Filter):

    def __init__(self, cut=(50, 50, 20)):
        if not hasattr(cut, '__iter__'):
            cut = [cut, cut, cut]

        self._cut = cut

    def apply(self, image):
        cut = self._cut
        if image.ndim < 3:
            raise ValueError('Provide 3D image.')

        f = numpy.fft.fft2(image, axes=(0, 1, 2))
        fshift = numpy.fft.fftshift(f)
        X, Y, Z = tuple(map(lambda x: x // 2, image.shape))
        fshift[X - cut[0]:X + cut[0], Y - cut[1]:Y + cut[1], Z - cut[2]:Z + cut[2]] = 0
        f_ishift = numpy.fft.ifftshift(fshift)

        filtered = numpy.abs(f_ishift)

        return filtered


def main():
    arr = numpy.ones((10, 10, 10))
    f = LoG()
    out = f.convolve(arr)
    print(out)
