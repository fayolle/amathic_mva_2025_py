import numpy as np


def to8U(img):
    if img.dtype == np.uint8:
        return img

    img = np.clip(img, 0.0, 1.0)
    return np.round(255.0*img).astype(np.uint8)


def to64F(img):
    if img.dtype == np.float64:
        return img
    return (1.0 / 255.0) * np.float64(img)
