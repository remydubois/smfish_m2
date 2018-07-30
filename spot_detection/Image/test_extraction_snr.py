from skimage import io
from filters import *
from vizu import *
from fitters import Gaussian, Mixture
from segmentation import *
from utils import *
from inspect import signature
from spotdetector import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from itertools import product
import warnings
from spot import Spot
import time
from copy import deepcopy


# warnings.simplefilter()


# @chain_methods
@extend_class(show_image)
@extend_class(show_detection_profile)
@extend_class(show_spots)
@extend_class(show_model)
@extend_class(show_cells)
class FQimage(object):
    """
    Main class for spot detection and image filtering, including SNR metric and detection profile analysis.
    Images are shaped in the right order: (x, y, z).
    The class instance has to hold both raw and filtered image. As of now, each image, once loaded, weights around
    350 Mb, and once filtered, around 700Mb.
    MIP views of an image have therefore been moved to the plotting methods ( MIP is recomputed at each view, which
    slows down the process but lights down images because it is not an attribute).

    TODO: DONE finally re-order image dimension in the right order. mistake fixed which came from matplotlib.

    """

    def __init__(self, verbose=1, data=None):
        """
        Main class for spot detection and image filtering, including SNR metric and detection profile analysis.
        Images are shaped in the right order: (x, y, z).
        The class instance has to hold both raw and filtered image. As of now, each image, once loaded, weights around
        350 Mb, and once filtered, around 700Mb.
        MIP views of an image have therefore been moved to the plotting methods ( MIP is recomputed at each view, which
        slows down the process but lights down images because it is not an attribute).

        - Methods in the class are chained what means they have to be called in a precised order (filter before detecting
        spots for instance). The chain_methods decorator ensures that, plus automatically calls methods of the chain
        which have not been called yet.
        - Vizualisation methods have been moved to vizu.py file but are attached thanks to extend_class decorator.


        :param verbose: whether to print or not progress bars and summaries of the evolution.
        :return: None
        """
        self.image_raw = data
        self.image_filtered = None
        self.spots = []
        self.SNR = None
        self._verbose = verbose
        self.cells = None
        self.name = None
        self.mixture_model = None

    def load(self, path):
        """
        Loads the image from disk into RAM, returns image with axis in the natural order (deep last).

        TODO Might be interesting to investigate numpy.load with memorymap in the future.
        TODO DONE MIP might me moved into 'show' methods for better optimization over memory charge.

        :param path: path of the file.
        :return: None
        """
        im = io.imread(path, dtype=numpy.uint8)
        if len(im.shape) > 2:
            self.image_raw = numpy.swapaxes(im, 2, 0)
        else:
            self.image_raw = im
        self.name = path.lower().split('/')[-1].replace('.tif', '')

    def filter(self, op=GaussianFilter):
        """
        Filters by first convolving the background with a gaussian filter.
        Then substract the obtained image to the origin and finally re-filter with another
        Gaussian filter with a variance 10 times smaller. Variance specified in utils module.
        TODO To be implemented: DOG, LocalMean, and more.

        :param op: operator taken from filters.py file. Must be an instance of the Filter class. (i.e. implementing
        'convolve' or 'apply' methods.
        :return: None, the image_filtered attribute is loaded with the filtered image.
        """

        if self._verbose > 0:
            print("Filtering...")

        # Import from utils specified params.
        params = get_filtering_params()

        negative = self.image_raw - op(sigma=params['sigma_bgd']).convolve(self.image_raw)

        self.image_filtered = op(sigma=params['sigma_spots']).convolve(negative)

    def _detect_spots(self, detector=LocalMax, **kwargs):
        """
        DEPRECATED, replaced by detect_and_fit for simplicity and speed issues.

        Detect spots with a specified detector (from the spotdetector.py module)
        and the detection params from utils module.
        Spots are identified by their position, i.e. 'x.y.z'.

        :param detector: an instance of the SpotDetector or subclass, i.e. implementing a 'locate' method returning
        array of positions of spots.
        :return: None, the spots attribute is a dict filled with spots (name i.e.  their position 'x.y.z' and their
        actual positions.)
        """
        if self._verbose > 0:
            print("Detecting...", end="")

        spots = detector(**kwargs).locate(self.image_filtered)

        # Spots are identified by their position:
        self.spots = [Spot(tuple(s)) for s in spots]
        if self._verbose > 0:
            print('%i spots detected.' % len(self.spots))

    def get_sigma_psf(self):
        """
        TODO Should return the variance of the PSF in order to compute correctly the filters of the fiter method.
        """
        pass

    def fit_spots(self, spot_model=Mixture, kind='individual'):
        """
        DEPRECATED Jump to next paragraph.
        This method goes through all the detected spots and fit a specified spot_model separately to each of them.
        TODO DONE If a model can not be safely fit to the spot, then the spot is deprecated and deleted from the spots list.
        Spot_models are built in the fitters module.
        Extract_cube comes from utils module.

        A GMM from sklearn mixture model is fit to the dataset. To do so (and avoid too large dataset) the pixel values
        are bucketized:
        X_train will be constituted of [x, y, z] times image_raw[x, y, z] for all the x, y, z. For obvious complexity
        reasons only points neighboring a spot are added to X_train are their value do not flow between 0 and 2^16-1
        because that would make a huge X_train.
        Even if this seems counter productive, it is much faster to do this rather than fitting a mixture of GMM
        density functions on the image because by doing as below, I can focus on spots whereas the other way I would
        have to fit the ENTIRE space which takes ages.
        Here we get a better estimation of the spots position (local peak max is free of computation time).


        :param spot_model: an instance of the SpotModel class or children from the fitters.py module, i.e. implementing
        a 'fit' method and showing a 'method' attribute.
        """

        model = spot_model()
        # print(model)

        # if model.kind == 'individual':
        #
        #     loop = self.spots
        #
        #     # to_delete = []
        #     if self._verbose > 0:
        #         loop = tqdm.tqdm(loop, desc="Fitting spot models...")
        #
        #     to_delete = []
        #     for k in loop:
        #         spot = self.image_filtered[extract_cube(point=k.coordinates, side=get_focus_size())]
        #         centers = [get_focus_size() // 2, ] * 3
        #         results = model.fit(centers=centers, data=spot)
        #
        #         # Filter spots for which a model could not be fit.
        #         if results:
        #             model.params = list(k.coordinates) + list(model.params)
        #             k.model = model
        #         else:
        #             to_delete.append(k)
        #
        #     # Filter spots and store in dict
        #     self.spots = [k for k in self.spots if k not in to_delete]
        #
        #     self.mixture_model = lambda x, y, z: sum([s.model.function(*s.model.params)(x, y, z) for s in self.spots])

        if kind == 'collective':
            mask = numpy.zeros(self.image_filtered.shape)
            for s in self.spots:
                mask[ellipse_in_shape(mask.shape, s.coordinates, (10, 10, 5))] = 1
            mask = mask.astype(bool)
            results = model.fit(centers=[s.coordinates for s in self.spots], data=self.image_filtered, mask=mask)

            if results:
                params = model.params.reshape(-1, 4)
                for s, p in zip(self.spots, params):
                    s.model = Gaussian()
                    s.model.params = p
                print(model.params)
                centers = [s.coordinates for s in self.spots]
                backgrounds = [[0], ] * len(self.spots)
                print(centers)
                print(backgrounds)
                self.mixture_model = model.function

        if self._verbose > 0:
            time.sleep(0.1)
            print('%i spots fit.' % len(self.spots))

    def detect_and_fit(self,
                       detector=DoG,
                       min_sigma=1,
                       max_sigma=5,
                       sigma_ratio=1.3,
                       threshold=0.01,
                       background_kernel=(30, 30, 10)):

        valid = ['DoG', 'LoG', 'DoH']
        if detector.__name__ not in valid:
            raise ValueError('Detector not adapted, use one of DoG, LoG, DoH.')

        if self._verbose > 0:
            print("\nDetecting...", end="", flush=True)

        # Get the background signal by smoothing with a much larger kernel than the spot width.

        blobs = detector(
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=sigma_ratio,
            threshold=threshold).locate(self.image_raw)

        self.spots = [Spot(tuple(c.astype(int))) for c in blobs[:, :3]]
        if self._verbose > 0:
            print('%i spots detected.' % blobs.shape[0])

        sigmas = blobs[:, 3]

        if self._verbose > 0:
            print("\nFitting...", end="", flush=True)
        for s, p in zip(self.spots, sigmas):
            s.model = Gaussian()
            ex = self.image_raw[extract_cube(s.coordinates, side=background_kernel)]
            background = numpy.mean(ex)
            p = itertools.chain(s.coordinates,
                                [background],
                                [self.image_raw[s.coordinates] - background],
                                [p, p, p / max(self.image_raw.shape[:2])])
            p_names = signature(s.model.function).parameters.keys()
            s.model.params = {k: v for (k, v) in zip(p_names, p)}
        if self._verbose > 0:
            print("fit.")

        funcs = [s.model.function(*s.model.params.values()) for s in self.spots]

        self.mixture_model = lambda x, y, z: sum([f(x, y, z) for f in funcs])

    def compute_snr(self):
        """
        The SNR is computed by comparing:
            - value at any point which does not belong to a spot zone. A spot zone being defined
            as the minimum distance between two spots (cf spot detector).
            TODO DONE In the future, it might be interesting to define a 'non-spotted' zone as the ensemble of points which
            are distant of more than 2*sigma_spot to the considered spot.
            - average amplitude of the spots in the cell.
            TODO Refactor the terrible loop in every ways.
            TODO Check the division: what happens when dist is odd ?

        In details, the code is fairly ugly but seems fine as the scope is reduced to the region of interest (cube
        surrounding a spot) and thereafter, all the computations are performed in this reduced scope.

        :param min_distance_appart_spots: the minimum distance for considering two spots appart. Should be greater than
        utils.get_focus_size()
        :return: list of snrs and their mean.

        Debugging: if snrs is empty (traceback to numpy.mean) then probably no is computed for any spot. Check sigma
        ceiling (beginning of for loop) and min_distance.
        """
        # Prune blobs with overlapping vicinities:
        blobs = numpy.array(
            [list(map(s.model.params.__getitem__, ['center_x', 'center_y', 'center_z', 'width_x'])) for s in
             self.spots])

        pruned_blobs = prune_blobs(blobs, overlap=0.)

        # Compute background
        background_signal = self.image_raw.copy()
        loop = pruned_blobs[10:11]
        if self._verbose > 0:
            loop = tqdm.tqdm(loop, desc="Computing SNRs for each spot")

        snrs = []
        shape = self.image_raw.shape
        for s in loop:
            t = time.time()
            environment = self.image_raw.copy()
            environment2 = self.image_raw.copy()
            print(time.time() - t)

            t = time.time()
            spot = ellipse_in_shape(shape, s[:3],
                                           (2 * s[-1], 2 * s[-1], 2 * s[-1] * shape[2] / max(shape[0], shape[1])))
            f, (ax0, ax1) = plt.subplots(nrows=2)
            environment[spot] = environment.max()
            cube = extract_cube((int(s[0]), int(s[1]), int(s[2])), side=20)
            ex = environment[cube]
            ax0.imshow(numpy.amax(ex, 2))
            print(time.time() - t)
            t = time.time()
            vicinity = ellipse_in_shape(shape, s[:3],
                                           (4 * s[-1], 4 * s[-1], 4 * s[-1] * shape[2] / max(shape[0], shape[1])))
            noise = filter(lambda t: t not in zip(*spot), zip(*vicinity))
            environment2[tuple(map(numpy.array, zip(*noise)))] = environment.max()
            ax1.imshow(numpy.amax(environment2[extract_cube((int(s[0]), int(s[1]), int(s[2])), side=20)], 2))
            f.show()
            print(sum(1 for _ in noise))
            print(time.time() - t)

            # get the spot signal
            t = time.time()
            spot_signal = self.image_raw[spot]
            print(time.time() - t)

            # Now filter it out from the environment
            t = time.time()
            environment[spot] = -1
            # noise_signal = self.image_raw[tuple(map(numpy.array, zip(*noise)))]
            noise_signal = environment[vicinity][environment[vicinity] > 0].ravel()
            print(noise_signal.shape)
            print(time.time() - t)


            # Now compute the SNR
            t = time.time()
            mean_noise = numpy.mean(noise_signal)
            energy_noise = numpy.std(noise_signal)
            energy_spot = numpy.sum((spot_signal - mean_noise) ** 2) / spot_signal.shape[0]
            print(time.time() - t)

            snr = energy_spot / energy_noise

            snrs.append(snr)

        if len(snrs) == 0:
            print('No SNR computed.')
            self.SNR = None
        else:
            self.SNR = round(numpy.mean(snrs), 1)

    def segment(self, sg=BasicSegmenter()):
        """
        This method is intended at segmenting the cells in FQImage on Mask images (not FISH). However basic, it seems
        to give a rather good
        approximation. The workflow is MIP -> local grad -> Otsu thresholding -> Connected components labelling ->
        Filtering components based on their size (using either hand-threshold or KMeans to distinguish actual cells
        from noise components.
        :param sg: A segmenter object.
        :return: None.
        """

        self.cells = sg.method(self.image_raw)

    def split(self):
        """
        Simply counting the number of spots per
        label does not work properly as some region do not have closed boundaries (for instance the segmented boundary
        has a 'C' shape but the cell is round. In this case spots will not be labeled as belonging to the 'C' region
        but they actually belong to the underlying cell which is badly segmented.

        DEPRECATED

        :param :
        :return:


        """
        sub_images = []

        for region in regionprops(self.cells):
            minr, minc, maxr, maxc = region.bbox
            sub_image = self.image_raw[max(0, minr - 10):maxr, max(0, minc - 10):maxc, :]

            sub_images.append(FQimage(data=sub_image))

        return sub_images

    def assign(self):
        """
        assign spots and models and stuff to each sub-cell sgemented within the mother image.
        :return: None

        """

        for s in self.spots:
            if self.cells[s[:2]] == 0:
                label = find_nearest_region(self.cells, *s[:2])
            else:
                label = self.cells[s[:2]]

            s.region = label


if __name__ == '__main__':
    im = FQimage(verbose=0)
    im.load(get_image_path())
    s = time.time()
    im.detect_and_fit(threshold=0.01, background_kernel=(100, 100, 5))
    # im.show_spots()
    im.compute_snr()
    # print(time.time() - s)
    print(im.SNR)
    # im.show_model()
    # print(im.spots[:5])
