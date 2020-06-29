"""
Visualization utils.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches


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
