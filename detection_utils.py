"""
Utils to load and predict with a object detection model bubild with the tensorflow object detection API.
"""

from pathlib import Path
import tensorflow as tf
import numpy as np
import bbox_utils as box


def load_model(model_name):

    model_dir = Path(model_name)/"saved_model"

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
