"""
Module to evaluate Multiple Object Tracking.

TODO:
    MOTA
    MOTP
"""

import sys
from pathlib import Path
import json
import numpy as np
from scipy.spatial import distance_matrix
from munkres import Munkres


def compute_assignment(cost_matrix, munkres):
    """
    Calculate the Munkres solution to the classical assignment problem.
    :param cost_matrix: (NxM matrix) Cost (or distance) matrix.
                        Rows should contain
                        Values with -1 are considered ~INF dist.
    :param munkres: instance of Munkres class
    :return: (list of tuples) Pairs minimizing distance
    """
    cost_matrix = np.array(cost_matrix)

    if np.any((cost_matrix != -1) & (cost_matrix < 0)):
        raise ValueError("cost_matrix can't contain negative values except -1!")

    # Assign max value to all values == -1
    max_value = np.max(cost_matrix) * 1e6
    cost_matrix = np.where(cost_matrix == -1, max_value, cost_matrix).astype(cost_matrix.dtype)

    # Munkres only handles square and rectangular matrices. It does *not* handle irregular matrices!
    # As a workaround the matrix is transposed and corresponding pairs are flipped.
    h, w = cost_matrix.shape
    if h > w:
        low_cost_pairs = np.array(munkres.compute(cost_matrix.T))[:, ::-1]
    else:
        low_cost_pairs = np.array(munkres.compute(cost_matrix))

    return low_cost_pairs


def remove_uninformative_rows(array, uninformative_value=-1):
    informative_rows = np.any(array != uninformative_value, axis=1)
    return array[~informative_rows, :], informative_rows


def sparse_rows_to_original(s, original_shape, row_with_content, fill=-1):
    """Add rows if there was content before making s sparse."""
    res = np.ones(original_shape) * fill

    row_idx = row_with_content.cumsum()
    for i, row in enumerate(row_with_content):
        if row:
            idx = row_idx[i] - 1
            res[i, ...] = s[idx]
    return res


class MOTEvaluator:
    """Evaluation of Multiple Object Tracking."""

    def __init__(self, groundtruth, hypotheses, distance_threshold):
        """

        :param groundtruth: (str|Path) groundtruth JSON file
        :param hypotheses: (str|Path) path to hypotheses JSON file
        :param distance_threshold: (int) max distance (in pixel) between prediction and truth to define true positive
        """
        self.groundtruth = JSONTracks(groundtruth)
        self.hypotheses = JSONTracks(hypotheses)
        self.distance_threshold = distance_threshold
        self.munkres_max = sys.maxsize
        self.munkres = Munkres()
        self.mapping = []
        self.mismatches = 0
        self.mismatches1 = 0
        self.mismatches2 = 0

    def evaluate(self):
        if len(self.groundtruth) != len(self.hypotheses):
            raise ValueError(f"Groundtruth number frames ({len(self.groundtruth)}) does not match"
                             f" Hypotheses number of frames ({len(self.hypotheses)}).")

        for i in range(len(self.groundtruth)):

            hy_coords, hy_source, hy_target, hy_id, hy_frame = self.hypotheses[i]
            gt_coords, gt_source, gt_target, gt_id, gt_frame = self.groundtruth[i]

            # MOTP Step 1: Find best assignments between all spots using the hungarian algorithm
            distances = distance_matrix(hy_coords, gt_coords)
            distances = np.where(distances <= self.distance_threshold, distances, self.munkres_max)
            assignments = compute_assignment(distances, self.munkres)
            hy_assignments, gt_assignments = assignments[:, 0], assignments[:, 1]

            # MOTP Step 2: Count matches
            matches = np.where(distances[hy_assignments, gt_assignments] <= self.distance_threshold)[0]
            false_negatives = len(gt_coords) - len(matches)
            false_positives = len(hy_coords) - len(matches)

            # Step 3: Map tracks
            hy_matches = hy_id[hy_assignments[matches]]
            gt_matches = gt_id[gt_assignments[matches]]
            self.mapping.append(dict(list(zip(hy_matches, gt_matches))))

            # Step 4: Find track mismatches
            hy_valid_source = hy_source[hy_assignments[matches]]
            gt_valid_source = gt_source[gt_assignments[matches]]

            if len(self.mapping) > 1:
                previous_mapping = self.mapping[-2]

                for hy_s, gt_s in zip(hy_valid_source, gt_valid_source):
                    if (hy_s is None) or (gt_s is None):
                        self.mismatches1 += 1
                        continue
                    gt_mapped = previous_mapping.get(hy_s)
                    if gt_mapped != gt_s:
                        aa = type(gt_mapped)
                        aaa = type(gt_s)
                        aaaa = gt_mapped != gt_s
                        self.mismatches2 += 1


class JSONTracks:
    """Parse tracking JSON and give functionality for evaluation metric computation."""

    def __init__(self, json_path):
        self.json_path = Path(json_path)
        self.json = json.load(open(str(self.json_path), "r"))
        self.frames = self.json["Frames"]
        self._init_annotation()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        return self.annotations[item]

    def _init_annotation(self):
        """
        Initialize a mapping of spot id's and annotation for each frame.
        annotations -> frames -> (coordinates, sources, targets, ids)
        """
        annotations = []
        for frame in self.frames:
            coordinates, sources, targets, ids = [], [], [], []
            frame_id = set()
            for spot_id, spot_annot in frame.items():
                coordinates.append((spot_annot["x"], spot_annot["y"]))
                sources.append(spot_annot["source"])
                targets.append(spot_annot["target"])
                ids.append(spot_id)
                frame_id.add(spot_annot["frame"])
                if len(frame_id) != 1:
                    raise ValueError(f"Invalid frame number found in spot: {spot_id}")
            annotations.append((
                np.array(coordinates, dtype=np.float),
                np.array(sources, dtype=np.str),
                targets,
                np.array(ids, dtype=np.str),
                frame_id.pop()))
        self.annotations = annotations


debug_eval = MOTEvaluator(r"D:\trackmate_to_json\tracks.json", r"D:\trackmate_to_json\tracks.json", 50)
debug_eval.evaluate()
print("MM1", debug_eval.mismatches1)
print("MM2", debug_eval.mismatches2)
