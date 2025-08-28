import numpy as np

def boxfilter(img_in, neigh_size):
    """
    Implementation of the box filter. 

    Args:
        img_in (np.ndarray): The input 2D grayscale image.
        neigh_size (int): The size of the box neighborhood. Must be an odd number.

    Returns:
        np.ndarray: The filtered output image.
    """
    r = (neigh_size - 1) // 2
    h, w = img_in.shape
    img_out = np.zeros_like(img_in, dtype=np.float64)

    im_cum = np.cumsum(img_in, axis=0)
    img_out[0:r+1, :] = im_cum[r:2*r+1, :]
    img_out[r+1:h-r, :] = im_cum[2*r+1:h, :] - im_cum[0:h-2*r-1, :]
    im_cum_full_height = im_cum[-1, :]
    img_out[h-r:h, :] = im_cum_full_height - im_cum[h-2*r-1:h-r-1, :]
    im_cum = np.cumsum(img_out, axis=1)
    img_out[:, 0:r+1] = im_cum[:, r:2*r+1]
    img_out[:, r+1:w-r] = im_cum[:, 2*r+1:w] - im_cum[:, 0:w-2*r-1]
    im_cum_full_width = im_cum[:, -1]
    img_out[:, w-r:w] = np.expand_dims(im_cum_full_width, axis=1) - im_cum[:, w-2*r-1:w-r-1]

    return img_out
