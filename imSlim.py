import numpy as np
from skimage import exposure, color

from boxfilter import boxfilter


def guidedfilter(src, guide, radius=5, eps=0.01):
    '''
    src: filtering grayscale image
    guide: guide grayscale image
    radius: filter radius (default value: 5, same as MATLAB)
    eps: regularization coefficient (default value: 0.01, same as MATLAB)
    '''
    ones = np.ones_like(guide)
    N = boxfilter(ones, radius)

    mean_I = boxfilter(guide, radius) / N
    mean_p = boxfilter(src, radius) / N
    corr_I = boxfilter(guide*guide, radius) / N
    corr_Ip = boxfilter(guide*src, radius) / N

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = boxfilter(a, radius) / N
    mean_b = boxfilter(b, radius) / N

    q = mean_a * guide + mean_b
    return q


def imSlim(rgbIn, b):
    rgbIn2 = np.copy(rgbIn)
    maxc = np.max(rgbIn)

    if maxc > 0:
        rgbIn2[:, :, 0] = rgbIn[:, :, 0] / maxc
        rgbIn2[:, :, 1] = rgbIn[:, :, 1] / maxc
        rgbIn2[:, :, 2] = rgbIn[:, :, 2] / maxc
    else:
        pass

    hsv = color.rgb2hsv(rgbIn2)
    v = hsv[:, :, 2]
    maxV = maxc

    q = min(0.4 + 3.0 / (4.0 + maxV), (1.0 + maxV) / 2.0)
    v = v** q

    mv = np.mean(v)
    p = 1.0 - 0.2 * (0.5 + np.arctan(100 * mv - 5) / np.pi)
    r = 0.01

    u = guidedfilter(v, v, radius=5, eps=0.01)
    idx = (u < 0.0)
    u[idx] = 0.0 
    v = v / (u**p + r)

    v = np.minimum(v, 1.0)

    v_adapted = exposure.equalize_adapthist(v)

    hsv[:, :, 2] = (1 - b) * v + b * v_adapted
    hsv[:, :, 1] = hsv[:, :, 1] * 0.6

    rgbOut = color.hsv2rgb(hsv)
    return rgbOut
