# Describe here the fitting performed for measuring gaussian amplitudes
from scipy.optimize import leastsq, fmin, curve_fit
import numpy
from .utils import Gaussian3D, MixtureGaussian3D
import warnings
import time
from sklearn.linear_model import LinearRegression


class SpotModel(object):
    """
    Class of spot models used to fit the spots.
    """

    def __init__(self, optimizer=curve_fit):
        """

        :param optimizer: by default the leastsq optimizer from scipy.
        """

        self.optimizer = optimizer
        self.function = None
        self.params = None
        self.kind = None

    def fit(self, data, centers, mask=None):
        """
        Fit a function onto focus data.
        TODO DONE Enforce sigmas to be positive, or take its aboslute values.
        TODO DONE OK Make sure fitting of sigmas comes in the right order (still the issue of axis ordering).
        TODO DONE NOT OK Understand why sigma does not fit anymore. Problem is about data which is centered on a spot.
        Fixed by building a get_focus_Size function which is used in both fitting and plotting methods.

        :param data: the data on which to fit the function
        :param sigmas: initial sigmas as starting points of the optimizer.
        :return: Nothing, set the spot model object.
        """
        params = numpy.array([[numpy.max(data[0]),
                               *[3., 3., 1.]
                               ], ] * len(centers))

        backgrounds = numpy.array([[data.min()], ] * len(centers))

        # def loss_func(p, c, b, d):
        #
        #     loss = (self.function(c, b, p)(*tuple(map(lambda m: m[mask], numpy.indices(d.shape)))) - \
        #             d[mask].ravel()) * 1.e-2
        #     print('loss_func', loss)
        #     print('loss_func_sq', (loss ** 2).sum())
        #     return loss

        def func(x, *p):
            # print('.', end='', flush=True)
            return self.function(centers, backgrounds, *p)(x[0], x[1], x[2])

        t = tuple(map(lambda m: m[mask], numpy.indices(data.shape)))
        xdata = numpy.array(t)

        warnings.simplefilter('error')
        try:
            results, _ = self.optimizer(func,
                                     xdata=xdata,
                                     ydata=data[mask].ravel(),
                                     p0=params)
            print('res', results)

        except RuntimeWarning:
            print('out')
            return None
        warnings.simplefilter('default')


        # Take positive sigmas.
        for i in [-3, -2, -1]:
            results[i] = abs(results[i])

        # Filter out weird results
        if any([s > 100 for s in results[-3:]]):
            return None

        self.params = results
        self.function = self.function(centers, backgrounds, *results)

        return True

    def __repr__(self):
        msg = 'Spot model with pre-set optimizer %s. Model function %s.' % (
            self.optimizer.__name__, self.function.__name__)
        if self.params is not None:
            msg += '\nModel fit with params: \n"%s"' % ', '.join(map(str, self.params))
        return msg


class Gaussian(SpotModel):

    def __init__(self):
        """
        Gaussian spot model defined by a 3D gaussian function.
        """
        super(Gaussian, self).__init__()

        # From utils
        self.function = Gaussian3D

        self.kind = 'individual'


class Mixture(SpotModel):

    def __init__(self):
        """
        Mixture of Gaussian models
        """
        super(Mixture, self).__init__()

        self.function = MixtureGaussian3D

        self.kind = 'collective'
