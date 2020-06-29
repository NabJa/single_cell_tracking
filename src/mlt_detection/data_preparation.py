"""
Data preparation to train ResNet23 as in:
Rempfler, Markus, et al. "Cell lineage tracing in lens-free microscopy videos." 2017
"""

import argparse
import cv2
from pathlib import Path
import numpy as np
from src.tf_detection_api.detection_utils import tf_dataset_generator
from src.utils.bbox_utils import boxes_to_center_points


def gaussian_kernel(filter_size, sigma, mean):
    kx = cv2.getGaussianKernel(filter_size[0], sigma)
    ky = cv2.getGaussianKernel(filter_size[1], sigma)
    k = kx * np.transpose(ky)
    k *= (mean / np.max(k))
    return k


def rescale_min_max(values, nmin=0, nmax=1):
    """Rescale values to new nmin and nmax"""
    vmin, vmax = values.min(), values.max()
    scaled = (values - vmin)/(vmax - vmin)
    scaled = scaled * (nmax - nmin) + nmin
    return scaled


def add_kernel(img, k, x, y):
    """
    Add a kernel to an image
    :param img: (array) image with dimension (W, H) to add kernel on
    :param k:  (array) kernel with dimension (W, H)
    :param x: (int) x coordinate
    :param y: (int) y coordinate
    :return: image with added kernel k
    """
    h, w = img.shape
    kh, kw = k.shape

    if (kh % 2 != 0) or (kw % 2 != 0):
        raise ValueError(f"Kernel size must be even. Given kernel shape: {k.shape}")
    if (type(x) != int) or (type(y) != int):
        raise ValueError(f"Point coordinates must be integer. Given: x={type(x)}, y={type(y)}")

    dkh, dkw = kw // 2, kh // 2
    xstart, xend, ystart, yend = x - dkw, x + dkw, y - dkh, y + dkh

    # Crop kernel on edges
    if xstart < 0:
        k = k[:, abs(xstart):]
        xstart = 0
    if ystart < 0:
        k = k[abs(ystart):, :]
        ystart = 0
    if xend > w:
        xend = w
        k = k[:, :xend-xstart]
    if yend > h:
        yend = h
        k = k[:yend-ystart, :]

    img[ystart:yend, xstart:xend] = np.maximum(img[ystart:yend, xstart:xend], k)

    return img


def generate_gaussian_mask(img, pos, sigma=8, kernel_size=30):
    """Apply gaussian kernel on all positions pos in image img"""
    kernel = gaussian_kernel((kernel_size, kernel_size), sigma, 1)
    for x, y in pos:
        x, y = int(round(x, 0)), int(round(y, 0))
        img = add_kernel(img, kernel, x, y)
    return img


def record_to_probability_map(record, sigma=8, kernel_size=30, crop_shape=None):
    """Yield probability map and coordinates from a tf_record generator"""
    dataset = tf_dataset_generator(record)
    for data in dataset:
        img, img_cords = data["image"], boxes_to_center_points(data["bboxes"])

        prob_map = np.zeros_like(img, dtype=np.float32)
        prob_map = generate_gaussian_mask(prob_map, img_cords, sigma=sigma, kernel_size=kernel_size)

        if not (crop_shape is None):
            pm_crops, coords = crop_image_with_positions(prob_map, img_cords, crop_shape)
            crops, _ = crop_image_with_positions(img, img_cords, crop_shape)
            yield img, img_cords, prob_map, crops, pm_crops, coords
        yield img, img_cords, prob_map


def crop_image_with_positions(img, pos, crop_shape):
    """Crop image into max numver of crops crop_shape and recalculate positions pos"""
    ch, cw = (crop_shape, crop_shape) if type(crop_shape) is int else crop_shape
    ih, iw = img.shape

    rows = iw // cw
    cols = ih // ch

    crops = []
    crop_cords = []

    ymin = 0
    xmax, ymax = cw, ch
    for col in range(cols):
        xmin = 0
        xmax = cw
        for row in range(rows):

            # Crop image
            crop = img[ymin:ymax, xmin:xmax]

            # Get crop coordinates
            idx = np.where((pos[:, 0] <= ymax) & (pos[:, 0] >= ymin) &
                           (pos[:, 1] <= xmax) & (pos[:, 1] >= xmin))
            coords = pos[idx]

            # Rescale coordinates to crop
            x = coords[:, 1] - xmin
            y = coords[:, 0] - ymin

            crops.append(crop)
            crop_cords.append(np.array(list(zip(x, y))))

            xmin += cw
            xmax += cw
        ymin += ch
        ymax += ch

    return crops, crop_cords


def _file_path(x):
    x = Path(x)
    if not x.is_file():
        raise ValueError(f"Given path is not a file.\n\tGiven: {x}")
    return x


def _path(x):
    x = Path(x)
    if not x.exists():
        raise ValueError(f"Given path does not exist.\n\tGiven: {x}")
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO support other formats as input
    # parser.add_argument("-i", "--image_path", type=_path)
    # parser.add_argument("-c", "--coordinates", type=_path)
    parser.add_argument("-t", "--tf_record", type=_file_path, required=True,
                        help="tf_record file containing the fields: image, bboxes")
    parser.add_argument("-o", "--output", type=_path, required=True,
                        help="Output path")
    args = parser.parse_args()

    pm_transformer = record_to_probability_map(args.tf_record, kernel_size=30, sigma=8, crop_shape=(224, 224))
    for i, transformed in enumerate(pm_transformer):

        # Generate output path
        out_path = args.output / f"annot_image_{i}"
        out_path.mkdir(exist_ok=True)

        # Save original image, coordinates and probility map
        image, coordinates, probability_map = transformed[0], transformed[1], transformed[2]
        cv2.imwrite(str(out_path/"original_image.png"), image)
        cv2.imwrite(str(out_path/"original_pm.png"), probability_map*255)
        np.save(str(out_path/"coordinates.npy"), coordinates)

        # CROP IMAGES
        # Generate crop path
        crops_path = out_path/"crops"
        crops_path.mkdir(exist_ok=True)

        # Save crops and coordinates
        for j, (crop, pm_crop, cord) in enumerate(zip(*transformed[3:])):
            cv2.imwrite(str(crops_path / f"{j}_lf.png"), crop)
            cv2.imwrite(str(crops_path/f"{j}_pm.png"), pm_crop*255)
            np.save(str(crops_path/f"{j}_crop_coordinates.npy"), cord)