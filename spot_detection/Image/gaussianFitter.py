# From Hazen Babcock (ZhuangLab/storm-analysis/sa_library/gaussfit.py

import math
import numpy as np
import scipy
import scipy.optimize

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

sigma = 1.0


# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------


def fitAFunctionLS(data, params, fn):
    """
    Least Squares fitting.
    """
    result = params
    errorfunction = lambda p: np.ravel(fn(*p)(*np.indices(data.shape)) - data)
    good = True
    [result, cov_x, infodict, mesg, success] = scipy.optimize.leastsq(errorfunction, params, full_output=1, maxfev=500)
    err = errorfunction(result)
    err = scipy.sum(err * err)
    if (success < 1) or (success > 4):
        print("Fitting problem!", success, mesg)
        good = False
    return [result, good]


def fitAFunctionMLE(data, params, fn):
    """
    MLE fitting, following Laurence and Chromy.
    """
    result = params

    def errorFunction(p):
        fit = fn(*p)(*np.indices(data.shape))
        t1 = 2.0 * np.sum(fit - data)
        t2 = 2.0 * np.sum(data * np.log(fit / data))
        return t1 - t2

    good = True
    try:
        [result, fopt, iter, funcalls, warnflag] = scipy.optimize.fmin(errorFunction, params, full_output=1,
                                                                       maxiter=500, disp=False)
    except:
        warnflag = 1
    if (warnflag != 0):
        print("Fitting problem!")
        good = False
        return [result, good]


def symmetricGaussian2D(background, height, center_y, center_x, width):
    return lambda y, x: background + height * np.exp(
        -(((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2) * 2)


def symmetricGaussian3D(background, height, center_z, center_y, center_x, width_z, width_xy):
    return lambda z, y, x: background + height * np.exp(-(((center_x - x) / width_xy) ** 2 +
                                                          ((center_y - y) / width_xy) ** 2 +
                                                          ((center_z - z) / width_z) ** 2) * 2)


def fitSymmetricGaussian2D(data, sigma):
    """
    Data is assumed centered on the gaussian and of size roughly 2x the width.
    """
    params = [np.min(data),
              np.max(data),
              0.5 * data.shape[0],
              0.5 * data.shape[1],
              2.0 * sigma]
    return fitAFunctionLS(data, params, symmetricGaussian2D)


def fitSymmetricGaussian3D(data, sigma):
    """
    Data is assumed centered on the gaussian and of size roughly 2x the width.
    """
    params = [np.min(data),
              np.max(data),
              0.5 * data.shape[0],
              0.5 * data.shape[1],
              0.5 * data.shape[2],
              2.0 * sigma[0],
              2.0 * sigma[1]]

    [result, good] = fitAFunctionLS(data, params, symmetricGaussian3D)
    return result, good


def fitSymmetricGaussian3Dbatch(data, pos, crop, sigma):
    """
    Fit multiple positions in data as specified in pos
    """

    result_all = np.zeros((pos.shape[0], 13))
    img_crop_all = np.zeros((pos.shape[0], 2 * crop[0] + 1, 2 * crop[1] + 1, 2 * crop[2] + 1))

    for idx, pos_loop in enumerate(pos):
        # Crop image
        img_crop = data[pos_loop[0] - crop[0]:pos_loop[0] + crop[0] + 1,
                   pos_loop[1] - crop[1]:pos_loop[1] + crop[1] + 1,
                   pos_loop[2] - crop[2]:pos_loop[2] + crop[2] + 1]

        # Perform fit in 3D
        fit_result, fit_status = fitSymmetricGaussian3D(img_crop, sigma)

        # Correct for crop
        # correct_crop    = np.array([0, 0,*(pos_loop-crop),0,0])
        pos_fit_corr = fit_result[2:5] + pos_loop - crop

        # Summarize fit
        result_all[idx, :] = np.concatenate((pos_loop, pos_fit_corr, fit_result), axis=0)
        img_crop_all[idx, :, :, :] = img_crop

    return result_all, img_crop_all


def plotGaussian3Dbatchfit(param_all, img_crop_all, ind_plot):
    """
    Plot results of 3D Gaussian fit
    """

    fit_result = param_all[ind_plot, 6:]
    img_crop = img_crop_all[ind_plot, :, :, :]

    fit_img_fun = symmetricGaussian3D(*fit_result)
    fit_img = fit_img_fun(*np.indices(img_crop.shape))

    # Create figure and axes
    import matplotlib.pyplot as plt

    fig_detect, (ax1, ax2) = plt.subplots(1, 2)

    plt.subplot(231)
    plt.imshow(np.amax(img_crop, axis=0), cmap='magma')
    plt.plot(fit_result[4], fit_result[3], marker='o', linestyle='--', color='g', label='Square')
    plt.axis('off')
    plt.title('Raw image-XY')
    plt.colorbar()

    plt.subplot(232)
    plt.imshow(np.amax(img_crop, axis=1), cmap='magma')
    plt.plot(fit_result[4], fit_result[2], marker='o', linestyle='--', color='g', label='Square')
    plt.axis('off')
    plt.title('Raw image-XZ')
    plt.colorbar()

    plt.subplot(233)
    plt.imshow(np.amax(img_crop, axis=2), cmap='magma')
    plt.plot(fit_result[3], fit_result[2], marker='o', linestyle='--', color='g', label='Square')
    plt.axis('off')
    plt.title('Raw image-YZ')
    plt.colorbar()

    ax2 = plt.subplot(234)
    plt.imshow(np.amax(fit_img, axis=0), cmap='magma')
    plt.plot(fit_result[4], fit_result[3], marker='o', linestyle='--', color='g', label='Square')
    plt.axis('off')
    plt.title('Fit image-XY')
    plt.colorbar()

    ax2 = plt.subplot(235)
    plt.imshow(np.amax(fit_img, axis=1), cmap='magma')
    plt.plot(fit_result[4], fit_result[2], marker='o', linestyle='--', color='g', label='Square')
    plt.axis('off')
    plt.title('Fit image-XZ')
    plt.colorbar()

    ax2 = plt.subplot(236)
    plt.imshow(np.amax(fit_img, axis=2), cmap='magma')
    plt.plot(fit_result[3], fit_result[2], marker='o', linestyle='--', color='g', label='Square')
    plt.axis('off')
    plt.title('Fit image-YZ')
    plt.colorbar()


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    background = 100
    height = 500
    center_z = 3
    center_y = 2
    center_x = 3.5
    width_z = 3
    width_xy = 1.5

    # Create function
    f_test = symmetricGaussian3D(background, height, center_z, center_y, center_x, width_z, width_xy)
    img_gauss = f_test(*np.indices((9, 5, 7)))

    # Perform fit in 3D
    sigma = [1, 1]
    fit_result, fit_status = gaussFit.fitSymmetricGaussian3D(img_gauss, sigma)

    # Apply fit
    fit_img_fun = symmetricGaussian3D(*fit_result)
    fit_img = fit_img_fun(*np.indices(img_gauss.shape))
