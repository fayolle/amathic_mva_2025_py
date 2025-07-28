import numpy as np


def to8U(img):
    if img.dtype == np.uint8:
        return img
    return np.clip(np.uint8(255.0 * img), 0, 255)


def to64F(img):
    if img.dtype == np.float64:
        return img
    return (1.0 / 255.0) * np.float64(img)
