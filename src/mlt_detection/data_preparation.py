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


ANNOTATION_FOLDER_STRUCTURE = """
main_path
    -> cell_type_x
        -> experiment_id
            -> pattern
        -> experiment_id
        -> ...
    -> ...
"""


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
    if type(kernel_size) is int:
        kernel_size = (kernel_size, kernel_size)
    kernel = gaussian_kernel(kernel_size, sigma, 1)
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

        pm_crops, coords = crop_image_with_positions(prob_map, img_cords, crop_shape)
        crops, _ = crop_image_with_positions(img, img_cords, crop_shape)
        yield img, img_cords, prob_map, crops, pm_crops, coords


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


def transform_annotation(record, output):
    """Genrate ResNet23 type data from single tfrecord"""
    pm_transformer = record_to_probability_map(record, kernel_size=30, sigma=8, crop_shape=(224, 224))
    for i, transformed in enumerate(pm_transformer):

        # Generate output path
        out_path = output / f"annot_image_{i}"
        out_path.mkdir(exist_ok=True)

        # Save original image, coordinates and probility map
        image, coordinates, probability_map = transformed[0], transformed[1], transformed[2]
        cv2.imwrite(str(out_path / "original_image.tif"), image)
        cv2.imwrite(str(out_path / "original_pm.tif"), probability_map * 255)
        np.save(str(out_path / "coordinates.npy"), coordinates)

        # CROP IMAGES
        # Generate crop path
        crops_path = out_path / "crops"
        crops_path.mkdir(exist_ok=True)

        # Save crops and coordinates
        for j, (crop, pm_crop, cord) in enumerate(zip(*transformed[3:])):
            cv2.imwrite(str(crops_path / f"{j}_lf.tif"), crop)
            cv2.imwrite(str(crops_path / f"{j}_pm.tif"), pm_crop * 255)
            np.save(str(crops_path / f"{j}_crop_coordinates.npy"), cord)


def transform_all_annotations(main_path, pattern, output):
    """
    Generate ResNet23 training folder structure.
    :param main_path: (string|Path) Path to annotations folder. Folder structure:
                        main_path
                            -> cell_type_x
                                -> experiment_id
                                    -> pattern
                                -> experiment_id
                                -> ...
                            -> ...
    :param pattern: (string) Glob pattern of tfrecord to search for in folder structure. (e.g. \'lensfree*.tfrecord\')
    :param output: (string|Path) Path to save the output.
    :return: None
    """
    main_path, output = _dir_path(main_path), _dir_path(output)
    datasets = list(main_path.rglob(pattern))

    for i, dataset in enumerate(datasets):
        print(f"Transforming dataset {i}/{len(datasets)}: {dataset}", end="\r")
        for j, (_, _, _, crops, pm_crops, coords) in enumerate(record_to_probability_map(dataset, crop_shape=(224, 224))):
            for k, (cr, pm, co) in enumerate(zip(crops, pm_crops, coords)):
                crop_path = output/dataset.parent.name/f"image{j}_crop{k}"
                crop_path.mkdir(parents=True)
                cv2.imwrite(str(crop_path / "prob_map.tif"), pm)
                cv2.imwrite(str(crop_path/"image.tif"), cr)
                np.save(str(crop_path/"coordinates.npy"), co)


def _file_path(x):
    x = Path(x)
    if not x.is_file():
        raise ValueError(f"Given path is not a file.\n\tGiven: {x}")
    return x


def _dir_path(x):
    x = Path(x)
    if not x.is_dir():
        raise ValueError(f"Given path is not a directory.\n\tGiven: {x}")
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
    parser.add_argument("-t", "--tf_record", type=_file_path,
                        help="tf_record file containing the fields: image, bboxes")
    parser.add_argument("-d", "--data", type=_dir_path,
                        help=f"Dataset with dataset structure as described in ANNOTATION_FOLDER_STRUCTURE. Call "
                             f"--struct for description.")
    parser.add_argument("-p", "--pattern", type=str,
                        help="Glob pattern of data data to search for (e.g. lensfree.tfrecord).")
    parser.add_argument("-o", "--output", type=_dir_path,
                        help="Output path")
    parser.add_argument("--struct", action='store_true')
    args = parser.parse_args()

    if args.struct:
        print(ANNOTATION_FOLDER_STRUCTURE)
        exit()
    elif args.output is None:
        raise ValueError("Invalid arguments. --output is required!")
    elif args.tf_record is not None:
        transform_annotation(args.tf_record, args.output)
        exit()
    elif args.data is not None:
        if args.pattern is None:
            raise ValueError("Pattern required! ")
        transform_all_annotations(args.data, args.pattern, args.output)
        exit()
    else:
        raise ValueError("No valid Arguments given. Requieres one of the following arguments: --data, --tf_record")
