""" Image processing utilities
"""

import skimage.color
import skimage.transform


def rgb_to_luminance(img):
    return skimage.color.rgb2gray(img)


def resize_img(img, new_size):
    return skimage.transform.resize(img, new_size)

