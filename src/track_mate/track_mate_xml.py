"""
Base module to handle TrackMate XML file.
"""

import xmltodict
from collections import OrderedDict
from pathlib import Path
import json


class TrackMateXML:
    """Data structure of TrackMate's xml file."""

    def __init__(self, xml_path):
        self.xml_path = Path(xml_path)

        with open(xml_path) as file:
            xml_file = xmltodict.parse(file.read())

        self.xml = xml_file["TrackMate"]

        self.nspots = self.xml["Model"]["AllSpots"]["@nspots"]
        self.nframes = self.xml["Settings"]["ImageData"]["@nframes"]

        self._init_frames()
        self._init_tracks()

    def _init_tracks(self):
        for i, track in enumerate(self.xml["Model"]["AllTracks"]["Track"]):
            edges = track["Edge"]
            if type(edges) == list:
                for edge in edges:
                    self._update_source_targets(edge)
            elif type(edges) == OrderedDict:
                self._update_source_targets(edges)
            else:
                raise TypeError(f"Edges list of unknown type: {type(edges)} in track {i}.")

    def _update_source_targets(self, edge):
        source_id = edge["@SPOT_SOURCE_ID"]
        target_id = edge["@SPOT_TARGET_ID"]

        source_spot = self.spot_index[source_id]
        target_spot = self.spot_index[target_id]

        self.frames[source_spot.frame].spots[source_id].add_target(target_id)
        self.frames[target_spot.frame].spots[target_id].add_source(source_id)

    def _init_frames(self):
        self.frames = []
        self.spot_index = {}
        for f in self.xml["Model"]["AllSpots"]["SpotsInFrame"]:
            frame = Frame(f)
            for s in f["Spot"]:
                spot = Spot(s)
                frame.add_spot(spot)
                self.spot_index[spot.id] = spot
            self.frames.append(frame)

    def save_as_json(self, json_path=None):
        if json_path is None:
            json_path = self.xml_path.parent.joinpath(self.xml_path.stem + ".json")
        json_dict = {frame.number: {spot.id: spot.to_dict() for spot in frame} for frame in self.frames}
        json.dump(json_dict, open(str(json_path), "w"), sort_keys=True, indent=4)


class Frame:
    def __init__(self, f):
        self.number = f["@frame"]
        self.xy_coordinates = []
        self.spots = OrderedDict()

    def __getitem__(self, item):
        return list(self.spots.values())[item]

    def __len__(self):
        return len(self.spots)

    def add_spot(self, spot):
        self.xy_coordinates.append((spot.x, spot.y))
        self.spots[spot.id] = spot


class Spot:
    def __init__(self, s):
        self.id = s["@ID"]
        self.frame = int(s["@FRAME"])
        self.x = float(s["@POSITION_X"])
        self.y = float(s["@POSITION_Y"])
        self.source = []
        self.target = []

    def __str__(self):
        return f"ID={self.id}, X={self.x}, Y={self.y}, Source={self.source}, Target={self.target}"

    def add_source(self, sid):
        if len(self.source) == 0:
            self.source.append(sid)
        else:
            raise ValueError(f"Multiple source IDs in spot: {self.id}."
                             f"Existing source: {self.source}. Trying to add: {sid}")

    def add_target(self, tid):
        if len(self.source) < 2:
            self.target.append(tid)
        else:
            raise ValueError(f"More then two target IDs in spot{self.id}."
                             f"Existing target: {self.target}. Trying to add: {tid}")

    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "source": self.source,
            "target": self.target
        }
