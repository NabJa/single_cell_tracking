"""
Visualization utils.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import cv2

def plot_bboxes_on_image(image, *bbox_instances, bbox_format="xy1xy2", labels=None, title=""):
    """
    Plot bounding boxes on image.

    :image:
    :bbox_instances: Bboxes with format specified in bbox_format.
    :bbox_format: Format of how bbox is saved. E.g. xy1xy2 = (xmin, ymin, xmax, ymax)
    :labels: Legend labels for given bboxes.
    """

    colors = plt.get_cmap("Set1").colors

    assert len(bbox_instances) < len(colors), f"Only {len(colors)} bbox instances supported."

    fig, ax = plt.subplots(1)
    fig.set_size_inches(16, 16)
    ax.set_title(title)

    # Display the image
    ax.imshow(image, cmap="gray")

    legend_lines = []
    labels = labels or [str(i) for i in range(len(bbox_instances))]

    for i, bboxes in enumerate(bbox_instances):
        legend_lines.append(Line2D([0], [0], color=colors[i], lw=4))
        for bbox in bboxes:
            x, y, w, h = parse_bbox(bbox, bbox_format, "xywh")
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor=colors[i], facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

    ax.legend(legend_lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def parse_bbox(bbox, bbox_format, res_format="xywh"):
    """
    Restructure bbox to given format.

    :param bbox: bounding box with 4 coordinates
    :param bbox_format: Coordinate structure
            e.g. xywh = [upper_left_x, upper_left_y, width, height]
            or xy1xy2 = [upper_left_x, upper_left_y, lower_right_x, lower_right_y, ]
    :param res_format: Output format
    :return: Newly formatted bounding box
    """

    if bbox_format == "xywh":
        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
    elif bbox_format == "xy1xy2":
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
    elif bbox_format == "yx1yx2":
        ymin, xmin, ymax, xmax = bbox
        width = xmax - xmin
        height = ymax - ymin
    else:
        raise NotImplementedError(f"{bbox_format} not supported.")

    if res_format == "xywh":
        return xmin, ymin, width, height
    if res_format == "xy1xy2":
        return xmin, ymin, xmax, ymax
    if res_format == "yx1yx2":
        return ymin, xmin, ymax, xmax
    else:
        raise NotImplementedError(f"{res_format} not supported.")


def plot_spots_on_video(frame_annotation, image_dir):
    """
    Create generator to to yield TrackMate annotation visualized on all images in image_dir.

    :param frame_annotation: JSONTracks object (can be found in src.evaluation.mot_evaluation).
    :param image_dir: Path to folder containing images.
    :return: Generator yielding annotated images.
    """
    image_dir = Path(image_dir)
    image_paths = list(image_dir.iterdir())

    assert len(frame_annotation) == len(image_paths), \
        f"Unequal number of annotations ({len(frame_annotation)}) and images ({len(image_paths)})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    font_color = (255, 0, 0)  # BGR

    for annotation, image_path in zip(frame_annotation, image_paths):

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        for spot_id, spot_annot in annotation.items():
            pos = int(spot_annot["x"]), int(spot_annot["y"])
            image = cv2.putText(image, str(spot_id), pos, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            image = cv2.circle(image, pos, 5, font_color, -1)
            image = cv2.circle(image, pos, 5, (0, 0, 0), 1)  # Black edge around circle

        yield image


def write_video_from_generator(generator, output, fps=5):
    image = next(generator)
    height, width, *_ = image.shape
    video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    video.write(image)
    for image in generator:
        video.write(image)
    video.release()
