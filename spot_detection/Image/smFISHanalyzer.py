'''
Introduction
============

Module to collect various function to analyze smFISH images and detect mRNAs
in 3D.

Usage
=====

'''

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import skimage
from skimage import io, filters, feature, morphology
from IPython.display import display
import numpy as np
import time
from os import path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

__version__ = '0.0.1'
__author__ = 'Florian MUELLER'
__email___ = 'muellerf.research@gmail.com'


class FQimg():
    '''Base class for AnnotationImporter object.'''

    def __init__(self):

        self.img_raw = []
        self.img_raw_MIP = []
        self.filt = []
        self.filt_MIP = []
        self.param = {}

    def loadImage3D(self, file_name_full, flag_output=(False, False)):
        """
        Load image
        """
        img_raw = skimage.io.imread(file_name_full)
        img_raw_MIP = (np.amax(img_raw, axis=0))

        # Analyze file-name = used to save results later
        img_path_name, img_file_name = path.split(file_name_full)
        base_filename, ext = path.splitext(img_file_name)

        # Save parameters
        self.img_raw = img_raw
        self.img_raw_MIP = img_raw_MIP
        self.img_path_name = img_path_name
        self.img_file_name = img_path_name
        self.img_file_name_base = base_filename
        self.img_file_name_ext = ext

        # Show some output
        if flag_output[0] is True:
            print("## Size of loaded image (Z-Y-X): {}".format(img_raw.shape))

    def filterImage3D(self, flag_output=(False, False)):
        """
        Filter image
        """
        if not ('filter_name' in self.param):
            raise KeyError('Filter is not defined in parameters!')
            return

        if self.param['filter_name'] == "LoG":
            if flag_output[0] is True:
                print('## Filtering image with LoG filter ... ', end='')
            img_filt = skimage.filters.laplace(
                self.img_raw, ksize=self.param['ksize'])
            img_filt = skimage.img_as_uint(img_filt)

        elif self.param['filter_name'] == "2XG":
            if flag_output[0] is True:
                print('## Filtering image with double Gaussian filter ... ', end='')
            img_bgd = skimage.filters.gaussian(
                self.img_raw, self.param['filter_2XG_bgd'])
            img_filt = skimage.filters.gaussian(
                self.img_raw - img_bgd, self.param['filter_2XG_spot'])
            img_filt = skimage.img_as_uint(img_filt.astype('int'))
        else:
            raise NotImplementedError(
                'Filter method is not recognized: {}'.format(self.param['filter_name']))
            return

        if flag_output[0] is True:
            print(' done!')
        # Perform MIPS
        img_filt_MIP = (np.amax(img_filt, axis=0))

        # Save images
        self.img_filt = img_filt
        self.img_filt_MIP = img_filt_MIP

        # Show projections
        if flag_output[1] is True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(self.img_raw_MIP, cmap='magma')
            ax2.imshow(self.img_filt_MIP, cmap='magma')

    def saveImage(self, img_type):
        """
        Save different images.
        """

        if img_type == 'filt':
            # Get file-name
            file_name_save = path.join(
                self.img_path_name, self.img_file_name_base + "_filt" + self.img_file_name_ext)

            # Save renormalized image
            skimage.io.imsave(file_name_save, self.img_filt)
            print("Filtering image saved as")
            print(file_name_save)

    def detectSpots3D(self, flag_output=(False, False)):
        # Perform detection with local maximum - result is np array

        if not ('detect_method' in self.param):
            raise KeyError('No pre-detection method is defined!')
            return

        if self.param['detect_method'] == "LocMax":
            detect_pos = (skimage.feature.peak_local_max(self.img_filt, min_distance=self.param[
                'detect_mind_dist'], threshold_abs=self.param['detect_int'], threshold_rel=None))

            Ndet, dum = detect_pos.shape
            if flag_output[0] is True:
                print(
                    'Local Maximum detection - number of pre-detections: {}'.format(Ndet))

        else:
            raise NotImplementedError(
                'Pre-detection method is not recognized: {}!'.format(self.param['detect_method']))
            return

        # Get coordonates of pre-detected positions
        pos_detect_z = np.ndarray.tolist(detect_pos[:, 0])
        pos_detect_x = np.ndarray.tolist(detect_pos[:, 2])
        pos_detect_y = np.ndarray.tolist(detect_pos[:, 1])

        # Store information
        self.detect_pos = detect_pos
        self.detect_int = self.img_filt[
            pos_detect_z, pos_detect_y, pos_detect_x]

        if flag_output[1] is True:

            # Create figure and axes
            fig_detect, (ax1, ax2) = plt.subplots(1, 2)

            plt.subplot(121)
            plt.imshow(self.img_filt_MIP, cmap='magma')
            plt.axis('off')
            plt.title('Filtered image')

            ax2 = plt.subplot(122)
            plt.imshow(self.img_filt_MIP, cmap='magma')
            plt.axis('off')
            plt.title('Filtered with detection')

            # Create circle patches and add them to the image
            for x, y in zip(pos_detect_x, pos_detect_y):
                detect_circ = patches.Circle(
                    (x, y), linewidth=0.5, edgecolor='g', facecolor='none')
                ax2.add_patch(detect_circ)

            # # Save image results of detection
            # if flag_save is True:
            #     file_name_save = path.join(img_path_name, base_filename + "_detect.png")
            #     fig_detect.savefig(file_name_save, dpi=900,bbox_inches='tight')

    def fitSpots3D(self, flag_output=(False, False)):

        self.fit_result, self.img_crop_fit = gaussFit.fitSymmetricGaussian3Dbatch(
            self.img_raw, self.detect_pos, self.param['fit_crop'], self.param['fit_sigma_init'])
