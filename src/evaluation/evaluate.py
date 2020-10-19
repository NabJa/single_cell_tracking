"""
Evaluate tracks given JSONs (as in json_data.py) or TrackMate XMLs.

Requires py-motmetrics to be installed.
Used commit: https://github.com/cheind/py-motmetrics/tree/0a6b598d9c966b05c4317f051119948d9140d0e2
"""

import argparse
from pathlib import Path
import motmetrics as mm
from src.evaluation.json_data import JSONTracks


def evaluate(objects, hypothesis, verbose=True):
    obj_json = JSONTracks(objects, from_xml=(objects.suffix == ".xml"))
    hyp_json = JSONTracks(hypothesis, from_xml=(hypothesis.suffix == ".xml"))

    assert obj_json.nframes == hyp_json.nframes, \
        f"Objects (nframes={obj_json.nframes}) and Hypothesis (nframes={hyp_json.nframes}) " \
        f"must have equal number of frames."

    acc = mm.MOTAccumulator(auto_id=True)

    for i in range(obj_json.nframes):
        if verbose:
            print(f"\rAccumulatating annotations {i + 1}/{obj_json.nframes}.", end="")
        gt_coords, gt_ids = obj_json.get_frame_coordinates(i), obj_json.get_frame_ids(i)
        hy_coords, hy_ids = hyp_json.get_frame_coordinates(i), hyp_json.get_frame_ids(i)
        dis = mm.distances.norm2squared_matrix(gt_coords, hy_coords, 15)
        acc.update(gt_ids, hy_ids, dis)

    if verbose:
        print("\nComputing metrics...")
    mh = mm.metrics.create()
    summary = mh.compute_many([acc],
                              metrics=mm.metrics.motchallenge_metrics,
                              names=['value'])
    str_summary = mm.io.render_summary(summary,
                                       formatters=mh.formatters,
                                       namemap=mm.io.motchallenge_metric_names)

    if verbose:
        print(50*"#" + "\n" + str_summary + "\n" + 50*"#")
    return summary


def _file_format_path(x):
    x = Path(x)
    if not x.is_file():
        raise ValueError(f"{x} is not a file!")
    elif x.suffix not in [".xml", ".json"]:
        raise ValueError(f"File {x} is not XML or JSON. (Must end with .xml or .json!)")
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-obj", required=True, type=_file_format_path, help="Ground truth tracks (= objects).")
    parser.add_argument("-hyp", required=True, type=_file_format_path, help="Predicted tracks (= hypothesis).")
    args = parser.parse_args()

    _ = evaluate(args.obj, args.hyp)
