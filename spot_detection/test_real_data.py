from Image.image import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dir",
                    default='Opera_Conf/3D/B09_DYNC1H1/r02c09f02-ch1sk1fk1fl1.tif',
                    help='This argument should be used to precise the directory containing the images.'
                    )


def main(argv_):
    im_path = '/Users/remydubois/Desktop/Remy/_REMY/Opera_WF/3D/B09_DYNC1H1/r02c09f02-ch3sk1fk1fl1.tif'
    im = FQimage()

    im.load(im_path)
    im.segment()
    im.show_cells()
    plt.show()
    # im.show_image()
    # im.detect_and_fit(threshold=0.05)

    # im.show_spots(); plt.show()
    # im.compute_snr()
    # time.sleep(0.1)
    # print('\n', im.SNR)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
