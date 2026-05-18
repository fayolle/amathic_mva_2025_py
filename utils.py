import numpy as np


def to8U(img):
    if img.dtype == np.uint8:
        return img

    img = np.clip(img, 0.0, 1.0)
    return np.round(255.0*img).astype(np.uint8)


def to64F(img):
    """
    Convert image to float64.

    uint8  -> divide by 255
    uint16 -> divide by 65535
    float  -> cast to float64 without rescaling
    """
    img = np.asarray(img)

    if np.issubdtype(img.dtype, np.floating):
        return img.astype(np.float64, copy=False)

    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0

    if img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0

    raise TypeError(f"Unsupported image dtype: {img.dtype}")
