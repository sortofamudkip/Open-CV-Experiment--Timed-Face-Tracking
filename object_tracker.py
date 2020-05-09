# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from limitedqueue.limitedqueue import LimitedQueue
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import math
import time
import cv2
import datetime as dt
import re

# sample full usage: python object_tracker.py -p deploy.prototxt -m res10_300x300_ssd_iter_140000.caffemodel -r (50,70),(350,70),(300,300),(100,300)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser("Counts the amount of faces that stay in an ROI for T seconds.\n")
ap.add_argument("-p", "--prototxt", default="deploy.prototxt", help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel", help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--time", type=float, default=3.0, help="how long it takes to target")
ap.add_argument("-r", "--roi", help="ROI: use format '(x0,y0),(x1,y1),...,(xn,yn)' to select points.", default="(50,70),(350,70),(300,300),(100,300)")
args = vars(ap.parse_args())

# set up ROI
try:
	roi_str = args["roi"]
	points = re.findall(r"\((\d+),(\d+)\)", roi_str)
	points = np.array([(int(a[0]), int(a[1])) for a in points])
	if len(points) <= 2: assert 0
except Exception:
	print("Cannot parse ROI!")
	exit(0)

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


######### Counting for 3s
cur_ID_set  = set(); cur_time = dt.datetime.now()
next_ID_set = set(); next_time = 0

three_secs = set()
cur_streaks = {}
TIME_LIMIT = args["time"]

######### Crop dimensions
# x0, y0 = 50, 50
# x1, y1 = 250, 250

######### allow mouse control
boxes = []
def on_mouse(event, x, y, flags, params):
	angle = 0.5*math.pi - math.pi / 3.0  # 45 degrees
	if event == cv2.EVENT_LBUTTONDOWN:
		boxes.append((x,y)) # global
	elif event == cv2.EVENT_LBUTTONUP:
		x0, y0 = boxes[0]
		x1, y1 = x,y
		global points
		delta = int((y1-y0)*math.tan(angle))
		points = np.array([[x0, y0], [x1,y0], [x1-delta, y1], [x0+delta, y1]])
		boxes.clear()


# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	# original_frame = vs.read()
	# frame = original_frame[x0:x1, y0:y1]
	original_frame = vs.read()
	original_frame = imutils.resize(original_frame, width=400)

	# try to do a trapizoidal shape
	mask = np.zeros(original_frame.shape, dtype=np.uint8)
	roi_corners = np.array([points], dtype=np.int32)
	channel_count = original_frame.shape[2]
	ignore_mask_color = (255,)*channel_count
	cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
	frame = cv2.bitwise_and(original_frame, mask)


	# frame = original_frame[x0:x1, y0:y1]

	cv2.setMouseCallback('Frame', on_mouse, 0)

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(original_frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(original_frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(original_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# get all the people in the frame 
	IDs = {objectInfo[0] for objectInfo in objects.items()}
	next_ID_set = IDs # t_(i+1)
	next_time = dt.datetime.now()
	for ID in next_ID_set:
		if ID in cur_ID_set:
			delta_t = next_time - cur_time
			cur_streaks.setdefault(ID, dt.timedelta(0, 0, 0))
			cur_streaks[ID] += delta_t
		else:
			cur_streaks[ID] = dt.timedelta(0, 0, 0)
	# remove streaks of those no longer in set
	not_in_next = cur_ID_set - next_ID_set
	for ID in not_in_next:
		del cur_streaks[ID]

	three_secs = {ID for ID in cur_streaks if cur_streaks[ID].total_seconds() >= TIME_LIMIT}
	# print("People in frame longer than 3 seconds:", three_secs if three_secs else "None")

	cur_ID_set = next_ID_set
	cur_time = next_time

	cv2.putText(original_frame, "Number of people: {}".format(len(three_secs)), (50,10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

	strings = ["{}: {}s".format(c, cur_streaks[c].total_seconds()) for c in cur_streaks]
	for i, s in enumerate(strings):
		cv2.putText(original_frame, "Time of people: {}".format(s), (0,30+20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

	# cv2.rectangle(original_frame, (x0, y0), (x1, y1), (0,0,255), 3)
	cv2.polylines(original_frame, [points], True, (0,255,255), 3)


	# show the entire frame, and the ROI as well
	cv2.imshow("Frame", original_frame)

	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()