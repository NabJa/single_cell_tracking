"""
Wrapper to run https://github.com/cheind/py-motmetrics on custom JSON.
"""

from pathlib import Path
import numpy as np
import json
from src.track_mate.track_mate_xml import TrackMateXML


class JSONTracks:
    """Parse tracking JSON and give functionality for evaluation metric computation."""

    def __init__(self, path, from_xml=False):
        """
        :param path: Input path to JSON or XML.
        :param from_xml: Parse from XML if true
        """

        if from_xml:
            self.xml_path = Path(path)
            self.json = TrackMateXML(self.xml_path).generate_json_dict()
        else:
            self.json_path = Path(path)
            self.json = json.load(open(str(self.json_path), "r"))
        self.frames = self.json["Frames"]
        self.n_mother_spots = 0
        self.nspots = 0
        self.nframes = len(self.frames)
        self.annotated_frames = []
        self.unique_ids = set()
        self._init_annotation()

    def __len__(self):
        return len(self.annotated_frames)

    def __getitem__(self, item):
        return self.annotated_frames[item]

    def get_frame_coordinates(self, frame_number):
        return np.array([(spot["x"], spot["y"]) for spot in self.annotated_frames[frame_number].values()])

    def get_frame_ids(self, frame_number):
        return list(self.annotated_frames[frame_number].keys())

    def get_frame_sources(self, frame_number):
        frame = self.annotated_frames[frame_number]
        return [i["source"] for i in frame.items()]

    def get_frame_targets(self, frame_number):
        frame = self.annotated_frames[frame_number]
        return [i["target"] for i in frame.items()]

    def _init_annotation(self):
        """
        Annotate IDs with the following convention:
            1. Initialise every new spot with \"c[unique number]\"
            2. If new split: add a \"-1\" and \"-2\" to the child names
            3. If frame does not split. Keep same ID as in previous frame.
        """
        familiy_names = {}
        familiy_ids = {}

        for frame in self.frames:
            frame_annotation = {}
            for spot_id, spot in frame.items():

                # No mother? Welcome new mother cell!
                if spot["source"] is None:
                    self.n_mother_spots += 1
                    self.nspots += 1
                    spot_name = "c" + str(self.n_mother_spots)
                    unique_id = self.nspots

                # Mother still the same
                elif spot_id in familiy_names:
                    spot_name = familiy_names[spot_id]
                    unique_id = familiy_ids[spot_id]

                # New child
                elif spot_id in familiy_names:
                    child_names = familiy_names[spot_id]
                    spot_name = child_names.pop()
                    child_ids = familiy_ids[spot_id]
                    unique_id = child_ids.pop()
                else:
                    raise ValueError("Spot has no source!", spot_id)

                # Add spot to current frame annotation
                frame_annotation[unique_id] = {
                    "x": spot["x"],
                    "y": spot["y"],
                    "name": spot_name,
                    "source": spot["source"],
                    "target": spot["target"],
                    "xml_id": spot_id
                }

                # No children
                if spot["target"] is None:
                    continue

                # Save Children for later if any
                target_lenth = len(spot["target"])
                if target_lenth == 1:
                    familiy_names[spot["target"][0]] = spot_name
                    familiy_ids[spot["target"][0]] = unique_id
                elif target_lenth == 2:
                    familiy_names[spot["target"][0]] = spot_name + "-1"
                    familiy_names[spot["target"][1]] = spot_name + "-2"
                    familiy_ids[spot["target"][0]] = self.nspots + 1
                    familiy_ids[spot["target"][1]] = self.nspots + 2
                    self.nspots += 2
                else:
                    raise ValueError(f"Unsupported length of targets: len={target_lenth} in spot: {spot_id}")

            self.annotated_frames.append(frame_annotation)
