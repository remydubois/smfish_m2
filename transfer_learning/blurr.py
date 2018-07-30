# Blurring function for domain mixing
import numpy as np


def add_gaussian_noise(image):
    # Gently blurr
    scale = np.std(image) / 1.
    # Unbiased noise
    loc = 0
    noise = np.random.normal(size=image.shape, loc=loc, scale=scale)

    return noise + image
