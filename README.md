# Single Cell Tracking

This repository aims to use TrackMate and MLT to track single cell with given detections.
The following features will implemented:

* [x]  SSD and Faster R-CNN loading [here](tf_detection_api/detection_utils.py)
* [x]  ResNet23 model [here](src/mlt_detection/resnet23.py) (Reimplementation of *"Cell lineage tracing in lens-free microscopy videos." Rempfler, Markus, et al. 2017*)
* [x]  TrackMate integration [here](src/track_mate/generate_xml.py)
* [x]  Benchmarking of tracking methods [here](src/evaluation/evaluate.py)
* [ ]  MLT tracking

## Tracking evaluation
For a detailed explanation of tracking evaluation see [here](https://github.com/cheind/py-motmetrics/tree/0a6b598d9c966b05c4317f051119948d9140d0e2).
