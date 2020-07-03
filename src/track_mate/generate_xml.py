"""
Functions to transform predictions into TrackMate XML.
"""
import argparse

from os.path import join
from pathlib import Path

from collections import OrderedDict
from glob import glob
import pickle
import numpy as np
from tqdm import tqdm
import xmltodict
import cv2

from src.tf_detection_api import detection_utils


def prepare_template_xml(image_path, template_path="Template.xml"):

    # Using a minimal Template.xml located next to this scrpit as a basis
    # 'doc' will hold the XML-Tree as a dictionary
    with open(template_path) as template:
        doc = xmltodict.parse(template.read())

    # List images to be processed, filtered by ending
    images = list(Path(image_path).glob("*.png"))
    img = cv2.imread(str(images[0]), 0)

    height, width = img.shape[:2]

    # Add dimensions and number of images to ImageData and BasicSettings in XML
    doc['TrackMate']['Settings']['ImageData']['@filename'] = images[0]
    doc['TrackMate']['Settings']['ImageData']['@folder'] = str(image_path)
    doc['TrackMate']['Settings']['ImageData']['@width'] = str(width)
    doc['TrackMate']['Settings']['ImageData']['@height'] = str(height)
    doc['TrackMate']['Settings']['ImageData']['@nframes'] = str(len(images))

    # TODO sure about -1??
    doc['TrackMate']['Settings']['BasicSettings']['@xend'] = str(width)
    doc['TrackMate']['Settings']['BasicSettings']['@yend'] = str(height)
    doc['TrackMate']['Settings']['BasicSettings']['@tend'] = str(len(images))

    return doc


def predictions_to_xml(image_path, predout, xmlout, min_score=0.5):
    """
    Write a trackmate XML for given prediction file.

    :image_path:    path to image folder containing input images in png format.
    :predout:       path to pickled prediction file.
                    Format: {image_path: {detection_boxes:(array), detection_scores:(array)}}
    :xmlout:        path to write the xml
    :min_score:     min detection score for cutoff
    """

    doc = prepare_template_xml(image_path)

    predictions = pickle.load(open(str(predout), "rb"))

    for img_no, (image_path,  prediction) in enumerate(predictions.items()):
        detection_boxes = prediction["detection_boxes"]
        detection_score = prediction["detection_scores"]

        detection_boxes = detection_boxes[detection_score >= min_score]
        detection_score = detection_score[detection_score >= min_score]

        points = bboxes_to_points(detection_boxes)

        for cell_no, (point, score) in enumerate(zip(points, detection_score)):
            add_cell(doc, cell_id=f"{img_no:05d}{cell_no:05d}",
                     name=Path(image_path).name,
                     position_x=str(point[0]),
                     position_y=str(point[1]),
                     frame=img_no,
                     quality=score)

    with open(str(xmlout), 'w') as result_file:
        result_file.write(xmltodict.unparse(doc, pretty=True))


def bboxes_to_points(bboxes):
    """
    Takes bboxes in [ymin, xmin, ymax, xmax] format and transorms them to points in [x, y] format.

    :bboxes: [N, (ymin, xmin, ymax, xmax)] array of bboxes
    """
    ymins, xmins = bboxes[:, 0], bboxes[:, 1]
    ymaxs, xmaxs = bboxes[:, 2], bboxes[:, 3]

    widths = (xmaxs - xmins) / 2
    heights = (ymaxs - ymins) / 2

    x_coords = xmins+widths
    y_coords = ymins+heights

    points = np.stack((x_coords, y_coords), axis=-1)
    return points


def add_cell(xml, cell_id="", name="", position_x="", position_y="", frame=0, radius=15, quality=1):
    cell = cellToDict(cell_id, name, position_x, position_y, frame, radius, quality)

    n_spots = int(xml['TrackMate']['Model']['AllSpots']['@nspots'])

    if "SpotsInFrame" in xml['TrackMate']['Model']['AllSpots']:
        spots_in_frame = xml['TrackMate']['Model']['AllSpots']['SpotsInFrame']
        if isinstance(spots_in_frame, list):
            found = False
            for spot in spots_in_frame:
                if int(spot['@frame']) == frame:
                    spot['Spot'].append(cell)
                    found = True
                    break
            if not found:
                spots_in_frame.append(OrderedDict({
                    '@frame': str(frame),
                    'Spot': [cell]
                }))
        elif isinstance(spots_in_frame, OrderedDict):
            if spots_in_frame['@frame'] == str(frame):
                spots_in_frame['Spot'].append(cell)
            else:
                xml['TrackMate']['Model']['AllSpots']['SpotsInFrame'] = [spots_in_frame]
                xml['TrackMate']['Model']['AllSpots']['SpotsInFrame'].append(
                    OrderedDict({
                        '@frame': str(frame),
                        'Spot': [cell]
                    })
                )
    else:
        xml['TrackMate']['Model']['AllSpots']['SpotsInFrame'] = OrderedDict({
            "@frame": str(frame),
            "Spot": [cell]
        })

    xml['TrackMate']['Model']['AllSpots']['@nspots'] = str(n_spots + 1)


def cellToDict(ID, NAME, POSITION_X, POSITION_Y, FRAME, RADIUS, QUALITY):
    return OrderedDict({
        '@ID': ID,                          # Must be numeric
        '@name': NAME,                      # Can be any string
        '@QUALITY': str(QUALITY),
        '@POSITION_X': str(POSITION_X),
        '@POSITION_Y': str(POSITION_Y),
        '@POSITION_Z': "0.0",
        '@POSITION_T': str(FRAME),
        '@FRAME': str(FRAME),
        '@RADIUS': str(RADIUS),
        '@VISIBILITY': "1",
        '@MANUAL_COLOR': "-10921639",
        '@MEAN_INTENSITY': "255.00",
        '@MEDIAN_INTENSITY': "255.00",
        '@MIN_INTENSITY': "255.00",
        '@MAX_INTENSITY': "255.00",
        '@TOTAL_INTENSITY': "255.00",
        '@STANDARD_DEVIATION': "0.0",
        '@ESTIMATED_DIAMETER': str(RADIUS * 2),
        '@CONTRAST': "0.0",
        '@SNR': "1.0"
    })


def _dir_path(x):
    x = Path(x)
    if not x.is_dir():
        raise ValueError(f"Given path is not a directory: {x}")
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", "-i", required=True, type=_dir_path)
    parser.add_argument("--model_dir", "-m", required=True, type=_dir_path)
    parser.add_argument("--output_dir", "-o", required=True, type=lambda x: Path(x))
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"WARNING: Generating output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True)

    # If predictions do not exist, predict.
    if not args.output_dir.joinpath("predictions.p").is_file():
        detection_utils.save_predictions(args.image_dir, args.output_dir, args.model_dir, verbose=1)

    xml_out_name = args.output_dir/"trackmate.xml"
    predictions_to_xml(str(args.image_dir), args.output_dir/"predictions.p", str(xml_out_name))
