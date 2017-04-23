""" Image processing utilities
"""

import skimage.color
import skimage.transform
import numpy as np

def rgb_to_luminance(img):
    return skimage.color.rgb2gray(img)


def resize_img(img, new_size):
    return skimage.transform.resize(img, new_size)

def pg_preprocess(img):
    """ pre-processing 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    img = img[35:195]  # crop
    img = img[::2, ::2, 0]  # downsample by factor of 2
    img[img == 144] = 0  # erase background (background type 1)
    img[img == 109] = 0  # erase background (background type 2)
    img[img != 0] = 1  # everything else (paddles, ball) just set to 1
    return img.astype(np.float32).ravel()