# Detection and MOT

## Second assignment for computer vision course

This repository has two main task, perform pedestrian detection and perform MOT given the former detection.

### Repository structure

* motPapa is the package folder containing the relevant functions to perform the tasks
* demo folder contains 2 scripts *detections.py* and *tracking.py* , they perform the actual computations for this two task, with output in the output folder
* Metrics folder contains the relevant code to compute some metrics comparing ground truth data and the computed results
* Output folder will contain data about the compute metrics, the detections and tracking in realated named folders
* Data folder contains the given data (gt.txt, frames)

### Setup

* Optional: create a new virtual environment
* Install the required packages listed in requirements.txt  "pip install -r requirements.txt"
* please install the package, once placed in the folder: "pip install -e ."

### Run the Demos

#### Detection

Run the python script /demo/detection.py, at the end it will create a file /Output/detection/detections.csv

in the format <frame_id>, <x1>,<y1>, <x2>,<y2>.

### Tracking

Run the python script /demo/tracking.py, at the end it will create a file /Output/tracking/trackings.csv

in the format <frame_id>,<object_id>,<x_center>,<y_center>





