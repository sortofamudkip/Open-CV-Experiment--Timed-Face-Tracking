# Open CV Experiment: Timed Face Tracking
 
An application of [this](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/) tutorial regarding face recognition. This "fork" counts the number of people who stay in the ROI for longer than 3 seconds. The ROI can be changed by dragging the mouse across the interface.

## Environment

* Windows (preferred)
* Webcam reequired
* Python 3 with the following packages: `imutils`, `numpy` and `cv2`

## Usage

### Quick Start (all values set to default)
```
python object_tracker.py
```

### Options
* `-c` or `--confidence`: Confidence value of determining a human's face, as a float in range [0,1]. Default is 0.5.
* `-t` or `--time`: How long a face has to stay in ROI for the counter to count it, as a float larger than 0. Default is 3.0.
* `-r` or `--roi`: The ROI, determined by at least 3 points. The default is `(50,70),(350,70),(300,300),(100,300)`.
    * Input format: `(x0,y0),(x1,y1),(x2,y2),...,(xn,yn)`. **Do not put in spaces.**
    * Example: `--roi (50,70),(350,70),(300,300),(100,300)`
* `--prototxt` and `--model` do not need to be changed unless you want to specifically change the facial recognition program.

### Features

* Press `q` to exit the program. (Pressing X will not work.)
* You can drag the mouse along the interface to select the ROI. Currently it creates a trapezoid of 30° angles.
