"""
Utils to load and predict with a object detection model bubild with the tensorflow object detection API.
"""

import io
from pathlib import Path
import tensorflow as tf
import numpy as np
import src.utils.bbox_utils as box
from PIL import Image


def load_model(model_path):

    model_dir = Path(model_path)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def run_inference_for_single_image(model, image):

    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)

    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)
    detections_normalized = box.normalize_bboxes_to_image(output_dict["detection_boxes"], image)
    detections_normalized = np.apply_along_axis(box.bbox_yx1yx2_to_xy1xy2, 1, detections_normalized)

    output_dict["detection_boxes"] = detections_normalized

    return output_dict


def tf_dataset_generator(path):
    raw_image_dataset = tf.data.TFRecordDataset(str(path))

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset:

        image_filename = image_features['image/filename'].numpy()

        image_raw = image_features['image/encoded'].numpy()
        image = np.array(Image.open(io.BytesIO(image_raw)).convert('L'))

        height = np.array(image_features['image/height'].numpy())
        width = np.array(image_features['image/width'].numpy())

        xmin = np.array(image_features['image/object/bbox/xmin'].numpy()) * width
        ymin = np.array(image_features['image/object/bbox/ymin'].numpy()) * height
        xmax = np.array(image_features['image/object/bbox/xmax'].numpy()) * width
        ymax = np.array(image_features['image/object/bbox/ymax'].numpy()) * height
        bboxes = np.stack((xmin, ymin, xmax, ymax), axis=-1)

        yield {"name": image_filename, "image": image, "bboxes": bboxes}


def _parse_image_function(example_proto):

    image_feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing = True, default_value=0.0),
        'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing = True, default_value=0.0),
        'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing = True, default_value=0.0),
        'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing = True, default_value=0.0),
        'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing = True),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }

    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)
