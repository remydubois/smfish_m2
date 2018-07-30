# % matplotlib inline

from Image.image import *
# from scipy.ndimage.filters import maximum_filter
# warnings.simplefilter('ignore')
im_path = '/Users/remydubois/Desktop/Remy/_REMY/Opera_WF/3D/B09_DYNC1H1/r02c09f05-ch1sk1fk1fl1.tif'

im = FQimage()
im.load(im_path)
# from skimage.feature import blob_dog
raw = im.image_raw.copy()

image = img_as_float(raw)
k = int(log(float(4) / 1, 1.3)) + 1

# a geometric progression of standard deviations for gaussian kernels
sigma_list = numpy.array([1 * (1.3 ** i)
                          for i in range(k + 1)])

gaussian_images = [gaussian_filter(image, s) for s in sigma_list]
dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
              * sigma_list[i] for i in range(k)]

image_cube = numpy.stack(dog_images, axis=-1)
print(image_cube.shape)
print(image_cube.max())

thresh = gmm_threshold(numpy.amax(raw, 2), n_components=3, plot=True)
num_peaks = peak_local_max(raw)


f, ax = plt.subplots(ncols=2, figsize=(8, 16))
ax = ax.ravel()

ax[0].imshow(numpy.amax(image_cube, (2, 3)))

peaks = peak_local_max(image_cube, threshold_abs=image_cube.max() / 10, footprint=numpy.ones((8, 8, 35, 7), exclude_borders=False))
print(peaks.shape)
for p in peaks:
    c.add_patch(
        plt.Circle((p[1], p[0]), 5, fill=False, color='r'))
# # b.set_yscale('log')
# f.colorbar(cax)
f.show()