from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image


def _to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    # transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    assert (
        np.min(images) >= -1.0 - 1e-5
        and np.max(images) <= 1.0 + 1e-5
        and (images.dtype == np.float32 or images.dtype == np.float64)
    ), "The input images should be float64(32) and in the range of [-1.0, 1.0]!"
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.0) / 2.0 * (max_value - min_value) + min_value).astype(dtype)


def _im2uint(images):
    # transform images from [-1.0, 1.0] to uint8
    return _to_range(images, 0, 255, np.uint8)


def imread(path, mode="RGB"):
    """Read an image into [-1.0, 1.0] float64."""
    img = Image.open(path).convert(mode)
    arr = np.asarray(img)
    return arr / 127.5 - 1


def imwrite(image, path):
    # save an [-1.0, 1.0] image
    arr = _to_range(image, 0, 255, np.uint8)
    Image.fromarray(arr).save(path)


def imwriteShow(image, path):
    # save an [-1.0, 1.0] image
    arr = _to_range(image, 0, 255, np.uint8)
    img = Image.fromarray(arr)
    img.show()
    img.save(path)


_INTERP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "cubic": Image.Resampling.BICUBIC,
}


def imresize(image, size, interp="bilinear"):
    """Resize an image.

    Resize an [-1.0, 1.0] image.

    size: int, float or tuple
        * int - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image.

    interp: str, optional
        Interpolation to use for re-sizing.
    """

    uint8_img = _im2uint(image)

    pil = Image.fromarray(uint8_img)
    if isinstance(size, int):
        new_size = (int(pil.size[0] * (size / 100.0)), int(pil.size[1] * (size / 100.0)))
    elif isinstance(size, float):
        new_size = (int(pil.size[0] * size), int(pil.size[1] * size))
    else:
        # PIL expects (width, height)
        new_size = (int(size[1]), int(size[0]))

    resample = _INTERP.get(interp, Image.Resampling.BILINEAR)
    resized = pil.resize(new_size, resample=resample)
    return (np.asarray(resized) / 127.5 - 1).astype(image.dtype)


def immerge(images, row, col):
    """Merge images.

    merge images into an image with (row * h) * (col * w)

    `images` is in shape of N * H * W(* C=1 or 3)
    """
    if images.ndim == 4:
        c = images.shape[3]
    elif images.ndim == 3:
        c = 1

    h, w = images.shape[1], images.shape[2]
    if c > 1:
        img = np.zeros((h * row, w * col, c))
    else:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h : j * h + h, i * w : i * w + w, ...] = image

    return img
