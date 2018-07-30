from argparse import ArgumentParser
from multiprocessing import Process
from Image.tools import *
import threading
from skimage import img_as_ubyte
import pandas
import os
from Image import useable_functions
from Image.image import *
from functools import partial
from joblib import Parallel, delayed
import pickle

TESTDATA_PATH = '/Users/remydubois/Dropbox/Remy/Python/CodeFragments/TestData/'
REALDATA_PATH = '/Users/remydubois/Desktop/Remy/_REMY/hilo_tournant/Stacks/B09_DYNC1H1/'


class FishAnalyser(object):
    """
    Main Class for analyzing folders of images.
    """

    def __init__(self):
        pass

    @staticmethod
    def analyze(func, mother_queuer, jobs=3, write=True, **kwargs):
        """
        Method that goes through all the files contained in the pointed directory, retains the images and applied
        the SNR computation from FQimage.

        Computations are parallelized for speed improvement.

        TODO DONE The printing thread does not work that good (issue when a snr is put into the queue but the printing \\
        thread frequency is not high enough to catch this element and the queue is closed while not empty.
        TODO DONE I could probably replace all the flags by some task_done at least for local queues.

        :param dir, jobs, regex : the directory containing the images, number of jobs, regex to filter images name.
        :return: nothing, writes a txt file with snr computed for each image in the dir.
        """

        out = []
        output_queue = multiprocessing.JoinableQueue()

        global_queue, steps = mother_queuer(**kwargs)

        the_pool = [multiprocessing.Process(target=func, args=(global_queue, output_queue)) for _ in range(jobs)]

        print('%s files found, spawned on %i processes.' % (steps, jobs))
        global_start = time.time()

        for p in the_pool:
            p.start()

        print_thread = threading.Thread(target=pump_results, args=(out, output_queue, steps))
        print_thread.start()

        wait_for_all(*the_pool)

        output_queue.join()
        print_thread.join()

        for p in the_pool:
            p.join()

        global_time = time.time() - global_start

        print('\nTotal runtime %imin%isecs.\n' % (global_time // 60, int(global_time % 60)))

        if write:
            with open('%sinfo_%s.txt' % (dir, func.__name__), 'w+') as f:
                f.write('\n'.join(map(str, out)))

        return out


parser = ArgumentParser(description="Parser for reading target directory.")

parser.add_argument("--dir",
                    default=REALDATA_PATH,
                    help='This argument should be used to precise the directory containing the images.'
                    )

parser.add_argument("--n_jobs",
                    default=3,
                    type=int,
                    help='Number of workers to use.'
                    )

parser.add_argument("--regex",
                    default='.',
                    help='Regex to filter out images in a given file (usually the channel).')

parser.add_argument("--function",
                    default="compute_snr",
                    help='Name of the function to use, defined in useable_functions.py')

dirlist = [
    '/Users/remydubois/Desktop/Remy/_REMY/Opera_Conf/3D/B09_DYNC1H1/',
    '/Users/remydubois/Desktop/Remy/_REMY/Opera_Conf/3D/E09_MELK/',
    '/Users/remydubois/Desktop/Remy/_REMY/Opera_Conf/3D/E11_NDE1/'
    # '/Users/remydubois/Desktop/Remy/_REMY/Opera_WF/3D/B09_DYNC1H1/',
    # '/Users/remydubois/Desktop/Remy/_REMY/Opera_WF/3D/E09_MELK/',
    # '/Users/remydubois/Desktop/Remy/_REMY/Opera_WF/3D/E11_NDE1/'
    # '/Users/remydubois/Desktop/Remy/_REMY/hilo_tournant/Stacks/B09_DYNC1H1/',
    # '/Users/remydubois/Desktop/Remy/_REMY/hilo_tournant/Stacks/E09_MELK/',
    # '/Users/remydubois/Desktop/Remy/_REMY/hilo_tournant/Stacks/E11_NDE1/'
]


def seg(p):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        im = FQimage()
        im.load(p)
        im.segment()
        out = resize(im.cells, (512, 512), preserve_range=True).astype(numpy.uint8)
    return out


def mip(p):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        im = FQimage()
        im.load(p)
        mip = numpy.amax(im.image_raw, 2)
        out = resize(mip, (512, 512), preserve_range=True).astype(numpy.uint8)
        warnings.simplefilter('default')
    return out


def seg_and_mip(p):
    with warnings.catch_warnings():
        print('Treating %s...' % p, end="", flush=True)
        warnings.simplefilter('ignore')
        im = FQimage()
        im.load(p)
        im.segment()
        mip = numpy.amax(im.image_raw, 2)
        mip_out = resize(mip, (512, 512), preserve_range=True).astype(numpy.uint8)
        mask_out = resize(im.cells, (512, 512), preserve_range=True).astype(numpy.uint8)
        print('done.')

    return mip_out, mask_out


# def detect_and_seg(p1, p2):
#     """
#     There has to be a try-catch because some wells have a fish image but no dapi or vice-versa.
#
#     :param p1:
#     :param p2:
#     :return:
#     """
#     with warnings.catch_warnings():
#         # print('Treating %s...' % p1, end="", flush=True)
#         warnings.simplefilter('ignore')
#         fish = FQimage(verbose=0)
#         mask = FQimage(verbose=0)
#         try:
#             fish.load(p1)
#             mask.load(p2)
#         except FileNotFoundError:
#             return
#
#         microscope = p1.split('/')[6]
#         gene = p1.split('/')[8]
#
#         mask.segment()
#         # mask.show_cells()
#         cells = numpy.unique(mask.cells).shape[0] - 1  # dont forget background
#
#         heights = fish.detect_and_fit(threshold=0., return_profile=True)
#
#         # garbage
#         del mask
#         del fish
#
#         print('.', end="", flush=True)
#
#     return p1.split('/')[-1].lower().replace('.tif', ''), microscope, gene, heights, cells


# def main(args_):
#     for f in dirlist:
#         if not os.path.exists(f):
#             print(f)
#             raise OSError('not found.')
#
#     out = []
#     for idir in dirlist:
#         files = [f for f in listdir(idir)]
#
#         dapi = filter(lambda s: s.lower().endswith('.tif') and re.search('ch3' if 'hilo' not in idir else 'w3', s),
#                       files)
#         fish = filter(lambda s: s.lower().endswith('.tif') and re.search('ch1' if 'hilo' not in idir else 'w1', s),
#                       files)
#
#         # m = Parallel(n_jobs=4, verbose=11)(
#         #     delayed(detect_and_seg)(idir + p1, idir + p2) for (p1, p2) in zip(fish, dapi))
#         m = [detect_and_seg(idir + p1, idir + p2) for (p1, p2) in zip(fish, dapi)]
#         out.extend([t for t in m if t is not None])
#
#     df = pandas.DataFrame(out, columns=['image', 'microscope', 'gene', '#ARN', '#cells'])
#     df.to_pickle('/Users/remydubois/Desktop/Remy/_REMY/spot_detection/calibration/data.pkl')


def main(args_):

    def build_path_queue():
        files = [idir + f for idir in dirlist for f in listdir(idir)]
        fishes = filter(lambda s: s.lower().endswith('.tif') and re.search('ch1' if 'Opera' in s else 'w1', s), files)
        # masks = filter(lambda s: s.lower().endswith('.tif') and re.search('ch3' if 'Opera' in s else 'w3', s), files)

        path_queue = multiprocessing.JoinableQueue()
        steps = 0
        for p_fish in fishes:
            path_queue.put(p_fish)
            steps += 1

        # Let it constitute the queue. Just for safety.
        time.sleep(1)

        return path_queue, steps
        # return fishes, masks

    # f_p, m_p = build_path_queue()
    cytos = []
    nucs = []
    files = [idir + f for idir in dirlist for f in listdir(idir)]
    cms = filter(lambda s: s.lower().endswith('.tif') and re.search('ch2', s), files)
    dapis = filter(lambda s: s.lower().endswith('.tif') and re.search('ch3', s), files)

    for t in tqdm.tqdm(list(zip(dapis, cms))):
        try:
            cytoplasms, nucleis = useable_functions.seg_both(t)
            cytos.extend(cytoplasms)
            nucs.extend(nucleis)
        except:
            continue

    out = [cytos, nucs]
    with open('/Users/remydubois/Desktop/Remy/_REMY/Segmentation/data/sequences.pkl', 'wb+') as f:
        pickle.dump(out, f)



if __name__ == '__main__':
    args_ = parser.parse_args()

    main(args_)
