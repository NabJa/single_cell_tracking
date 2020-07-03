"""
Module to evaluate Multiple Object Tracking.

TODO:
    MOTA
    MOTP
"""

import numpy as np
from munkres import Munkres


def compute_assignment(cost_matrix):
    """
    Calculate the Munkres solution to the classical assignment problem.
    :param cost_matrix: (NxM matrix) Cost (or distance) matrix.
                        Rows should contain
                        Values with -1 are considered ~INF dist.
    :return: (list of tuples) Pairs minimizing distance
    """
    cost_matrix = np.array(cost_matrix)

    if np.any((cost_matrix != -1) & (cost_matrix < 0)):
        raise ValueError("cost_matrix can't contain negative values except -1!")

    # Assign max value to all values == -1
    max_value = np.max(cost_matrix) * 1e6
    cost_matrix = np.where(cost_matrix == -1, max_value, cost_matrix).astype(cost_matrix.dtype)

    m = Munkres()

    # Munkres only handles square and rectangular matrices. It does *not* handle irregular matrices!
    # As a workaround the matrix is transposed and corresponding pairs are flipped.
    h, w = cost_matrix.shape
    if h > w:
        low_cost_pairs = np.array(m.compute(cost_matrix.T))[:, ::-1]
    else:
        low_cost_pairs = np.array(m.compute(cost_matrix))

    return low_cost_pairs


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
        self.groundtruth = groundtruth
        self.hypotheses = hypotheses
        self.distance_threshold = distance_threshold
        self.munkres_default = -1
