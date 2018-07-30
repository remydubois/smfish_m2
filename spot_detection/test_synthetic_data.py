from Image.image import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dir",
                    default='Opera_Conf/3D/B09_DYNC1H1/r02c09f02-ch1sk1fk1fl1.tif',
                    help='This argument should be used to precise the directory containing the images.'
                    )


def main(argv_):

    feed_func = lambda x, y, z: Gaussian3D(200, 300, 25, 150, 3, 3, 3)(x, y, z) + \
                                Gaussian3D(300, 300, 25, 70, 4, 4, 4)(x, y, z) + \
                                Gaussian3D(150, 200, 25, 90, 2.5, 2.5, 2.5)(x, y, z) + \
                                Gaussian3D(50, 250, 25, 50, 3.5, 3.5, 3.5)(x, y, z) + \
                                Gaussian3D(300, 110, 25, 180, 1.5, 1.5, 1.5)(x, y, z) + \
                                Gaussian3D(450, 100, 25, 120, 3.5, 3.5, 3.5)(x, y, z)

    def inspect(loc, scale):
        data = numpy.fromfunction(feed_func, shape=(512, 512, 35)) + numpy.random.normal(loc=loc, scale=scale,
                                                                                         size=(512, 512, 35))
        im = FQimage(verbose=1)
        im.image_raw[0, 0, :] = 0
        heights = im.detect_and_fit(auto_level=False, return_profile=False, num_peaks=6)
        # f, (ax0, ax1) = plt.subplots(ncols=2)
        # ax0.plot(list(range(heights.shape[0])), heights)
        # ax1.plot(list(range(heights.shape[0])), numpy.gradient(numpy.gradient(heights)))
        # plt.show()
        # im.show_spots()
        im.compute_snr()
        time.sleep(0.1)
        print('SNR', im.SNR)

    inspect(30, 10)
    inspect(25, 9)
    inspect(20, 8)
    inspect(15, 7)
    inspect(10, 6)
    inspect(0, 0)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
