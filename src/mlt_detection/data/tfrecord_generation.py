"""
Transform ResNet23 data into tf record.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from src.utils import bbox_utils as bu


def object_detection_record_from_resnet_data(inp, out=None, ignore=None):
    inp = Path(inp)
    if out is None:
        out = str(inp/"resnet_data.tfrecord")

    print(f"Collecting data. Ignoring: {ignore} ...")
    if ignore is None:
        coord_path = [x for x in inp.rglob("coordinates.npy")]
        img_path = [x for x in inp.rglob("image.tif")]
    else:
        coord_path = [x for x in inp.rglob("coordinates.npy") if len(set(ignore).intersection(set(x.parts))) < 1]
        img_path = [x for x in inp.rglob("image.tif") if len(set(ignore).intersection(set(x.parts))) < 1]

    coords = [np.load(c) for c in coord_path]
    bboxes = [np.array([bu.point_to_box(p, 40, img_shape=(224, 224)) for p in points]) for points in coords]

    print(f"Writing record ...")
    write_detection_record(img_path, bboxes, out)
    print("FINISHED")


def image_pair_record_from_resnet_data(inp, image_name="image.tif", map_name="prob_map.tif", out=None, ignore=None):
    """
    Generate image pairs (microscopy image, probability map).

    :param inp: Root dir to search data from.
    :param image_name: Naming convention for microscopy images.
    :param map_name: Naming convention for probability map.
    :param out: Output path.
    :param ignore: List of folders to be ignored.
    """
    inp = Path(inp)
    if out is None:
        out = str(inp/"resnet_data.tfrecord")

    print(f"Collecting data. Ignoring: {ignore} ...")
    if ignore is None:
        img_path = [x for x in inp.rglob("image.tif")]
        pm_path = [x for x in inp.rglob("prob_map.tif")]
    else:
        img_path = [x for x in inp.rglob(image_name) if len(set(ignore).intersection(set(x.parts))) < 1]
        pm_path = [x for x in inp.rglob(map_name) if len(set(ignore).intersection(set(x.parts))) < 1]

    print(f"Writing record ...")
    write_pair_record(img_path, pm_path, out)
    print("FINISHED")


def write_pair_record(images, maps, filename):
    """
    Write resnet23 image pairs tf record to filename.

    :param images: List of image paths
    :param maps: List of probability map paths
    :param filename: Name of tf record file
    :return: None
    """

    with tf.io.TFRecordWriter(str(filename)) as writer:
        for i, (image_path, pm_path) in enumerate(zip(images, maps)):
            tf_example = pair_to_tf_example(image_path, pm_path)
            writer.write(tf_example.SerializeToString())
    print("Created tf-record file in: \n", filename)


def write_detection_record(images, bboxes, filename):
    """
    Write object detection record to filename.

    :param images: List of image paths
    :param bboxes: List of bboxes in format (xmin, ymin, xmax, ymax)
    :param filename: Name of tf record file
    :return: None
    """

    with tf.io.TFRecordWriter(str(filename)) as writer:

        for i, (image_path, bbox) in enumerate(zip(images, bboxes)):
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            tf_example = bbox_to_tf_example(image, image_path, bbox)
            writer.write(tf_example.SerializeToString())
    print("Created tf-record file in: \n", filename)


def encode_to_png(img):
    img = tf.cast(img, tf.uint8)
    if len(img.shape) == 2:
        return tf.image.encode_png(tf.expand_dims(img, axis=-1))
    elif len(img.shape) == 3:
        return tf.image.encode_png(img)
    else:
        raise ValueError(f"Invalid image shape {img.shape} in {img}")


def pair_to_tf_example(image_path, map_path):

    image_path, map_path = str(image_path), str(map_path)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    map_image = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)

    # TODO probability map needs different encoding, supporting floating points
    ecoded_image = encode_to_png(image)
    encoded_map = encode_to_png(map_image)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': _bytes_feature(image_path.encode("unicode_escape")),
        'image/encoded': _bytes_feature(ecoded_image),
        'map/filename': _bytes_feature(map_path.encode("unicode_escape")),
        'map/encoded': _bytes_feature(encoded_map),
    }))

    return tf_example


def bbox_to_tf_example(image, filename, bboxes):

    # Parse image metas
    height, width, *_ = image.shape
    image_ext = ".png"
    filename = str(filename)

    encoded_image = encode_to_png(image)

    # Parse bounding boxes
    xmins, xmaxs, ymins, ymaxs = [], [], [], []

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        xmins.append(x_min / width)
        xmaxs.append(x_max / width)
        ymins.append(y_min / height)
        ymaxs.append(y_max / height)

    classes_text = [b'Cell' for _ in bboxes]
    classes = [1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(filename.encode("unicode_escape")),
        'image/source_id': _bytes_feature(filename.encode("unicode_escape")),
        'image/encoded': _bytes_feature(encoded_image),
        'image/format': _bytes_feature(image_ext.encode("unicode_escape")),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/text': _bytes_list_feature(classes_text),
        'image/object/class/label': _int64_list_feature(classes),
    }))
    return tf_example


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input folder as generated by data_preparation.py")
    parser.add_argument("-o", "--output", help="Complete path to output record file. DEFAULT: input folder")
    parser.add_argument("--ignore", nargs="*", help="Ignore all annotations found containing any pattern provided."
                                                    "e.g. --ignore nrk_experiment a549_experiment")
    args = parser.parse_args()

    image_pair_record_from_resnet_data(args.input, args.output, args.ignore)
