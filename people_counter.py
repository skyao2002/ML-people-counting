# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import threading


# def incrementCount():
# 	count += 1
# def decrementCount():
# 	count -= 1

# initialize the list of class labels MobileNet SSD was trained to
# detect
class PeopleCounter(threading.Thread):
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	# default
	# prototxt='mobilenet_ssd/MobileNetSSD_deploy.prototxt', model='mobilenet_ssd/MobileNetSSD_deploy.caffemodel', input=None
	def __init__(self, threadID, name, **args):
		self.count = 0
		self.countChanged = True

		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.args = args
		# load our serialized model from disk
		print("[INFO] loading model...")
		self.net = cv2.dnn.readNetFromCaffe(self.args["prototxt"], self.args["model"])

		# if a video path was not supplied, grab a reference to the webcam
		if self.args["input"] == None:
			print("[INFO] starting video stream...")
			self.vs = VideoStream(src=0).start()
			time.sleep(2.0)

		# otherwise, grab a reference to the ip camera
		else:
			print("[INFO] opening ip camera feed...")
			self.vs = VideoStream(self.args["input"]).start()
			time.sleep(2.0)
	
	def run(self):
		# initialize the video writer (we'll instantiate later if need be)
		writer = None

		# initialize the frame dimensions (we'll set them as soon as we read
		# the first frame from the video)
		W = None
		H = None

		# instantiate our centroid tracker, then initialize a list to store
		# each of our dlib correlation trackers, followed by a dictionary to
		# map each unique object ID to a TrackableObject

		# base max distance is 50
		ct = CentroidTracker(maxDisappeared=40, maxDistance=100)
		trackers = []
		trackableObjects = {}

		# initialize the total number of frames processed thus far, along
		# with the total number of objects that have moved either up or down
		totalFrames = 0
		totalDown = 0
		totalUp = 0
		totalLeft = 0
		totalRight = 0

		# start the frames per second throughput estimator
		fps = FPS().start()
		temp = 0
		# loop over frames from the video stream
		while True:
			# grab the next frame and handle if we are reading from either
			# VideoCapture or VideoStream
			frame = self.vs.read()
			print(len(frame),len(frame[0]))
			temp += 1
			if temp == 10:
				break
			#frame = frame[1] if self.args.get("input", False) else frame

			# if we are viewing a video and we did not grab a frame then we
			# have reached the end of the video
			# if self.args["input"] is not None and frame is None:
			# 	break
			
			# resize the frame to have a maximum width of 500 pixels (the
			# less data we have, the faster we can process it), then convert
			# the frame from BGR to RGB for dlib
			frame = imutils.resize(frame, width=500)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# if the frame dimensions are empty, set them
			if W is None or H is None:
				(H, W) = frame.shape[:2]

			# if we are supposed to be writing a video to disk, initialize
			# the writer
			if self.args["output"] is not None and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(self.args["output"], fourcc, 30,
					(W, H), True)

			# initialize the current status along with our list of bounding
			# box rectangles returned by either (1) our object detector or
			# (2) the correlation trackers
			status = "Waiting"
			rects = []

			# check to see if we should run a more computationally expensive
			# object detection method to aid our tracker
			if totalFrames % self.args["skip_frames"] == 0:
				# set the status and initialize our new set of object trackers
				status = "Detecting"
				trackers = []

				# convert the frame to a blob and pass the blob through the
				# network and obtain the detections
				blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
				self.net.setInput(blob)
				detections = self.net.forward()

				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated
					# with the prediction
					confidence = detections[0, 0, i, 2]

					# filter out weak detections by requiring a minimum
					# confidence
					if confidence > self.args["confidence"]:
						# extract the index of the class label from the
						# detections list
						idx = int(detections[0, 0, i, 1])

						# if the class label is not a person, ignore it
						if self.CLASSES[idx] != "person":
							continue

						# compute the (x, y)-coordinates of the bounding box
						# for the object
						box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
						(startX, startY, endX, endY) = box.astype("int")

						# construct a dlib rectangle object from the bounding
						# box coordinates and then start the dlib correlation
						# tracker
						tracker = dlib.correlation_tracker()
						rect = dlib.rectangle(startX, startY, endX, endY)
						tracker.start_track(rgb, rect)

						# add the tracker to our list of trackers so we can
						# utilize it during skip frames
						trackers.append(tracker)

			# otherwise, we should utilize our object *trackers* rather than
			# object *detectors* to obtain a higher frame processing throughput
			else:
				# loop over the trackers
				for tracker in trackers:
					# set the status of our system to be 'tracking' rather
					# than 'waiting' or 'detecting'
					status = "Tracking"

					# update the tracker and grab the updated position
					tracker.update(rgb)
					pos = tracker.get_position()

					# unpack the position object
					startX = int(pos.left())
					startY = int(pos.top())
					endX = int(pos.right())
					endY = int(pos.bottom())

					# add the bounding box coordinates to the rectangles list
					rects.append((startX, startY, endX, endY))

			# draw a horizontal line in the center of the frame -- once an
			# object crosses this line we will determine whether they were
			# moving 'up' or 'down'
			if self.args["direction"] == "rightleft":
				cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 255, 255), 2)
			else:
				cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

			# use the centroid tracker to associate the (1) old object
			# centroids with (2) the newly computed object centroids
			objects = ct.update(rects)

			# loop over the tracked objects
			for (objectID, centroid) in objects.items():
				# check to see if a trackable object exists for the current
				# object ID
				to = trackableObjects.get(objectID, None)

				# if there is no existing trackable object, create one
				if to is None:
					if self.args["direction"] == "rightleft":
						if centroid[0] > W//2:
							to = TrackableObject(objectID, centroid, "right")
						else:
							to = TrackableObject(objectID, centroid, "left")
					else:
						if centroid[1] > H//2:
							to = TrackableObject(objectID, centroid, "down")
						else:
							to = TrackableObject(objectID, centroid, "up")

				# otherwise, there is a trackable object so we can utilize it
				# to determine direction
				else:
					if self.args["direction"] == "rightleft" and not to.counted:
						#print("current: {} side: {}".format(str(centroid[0]), to.side))
						if to.side == "right" and centroid[0] < W//2:
							to.side = "left"
							to.counted = True
							totalLeft += 1
							
							self.count = self.count + 1 if self.args["enter_direction"] == "left" else self.count - 1
							self.countChanged = True
							# incrementCount() if self.args["enter_direction" == "left"] else decrementCount()
						elif to.side == "left" and centroid[0] > W//2:
							to.side = "right"
							to.counted = True
							totalRight += 1

							self.count = self.count + 1 if self.args["enter_direction"] == "right" else self.count - 1
							self.countChanged = True

					elif self.args["direction"] == "updown" and not to.counted:
						#print("current: {} side: {}".format(str(centroid[0]), to.side))
						if to.side == "up" and centroid[1] > H//2:
							to.side = "down"
							to.counted = True
							totalDown += 1
						elif to.side == "down" and centroid[1] < H//2:
							to.side = "up"
							to.counted = True
							totalUp += 1

				# store the trackable object in our dictionary
				trackableObjects[objectID] = to

				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "ID {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

			# construct a tuple of information we will be displaying on the
			# frame
			if self.args["direction"] == "rightleft":
				info = [
					("Left", totalLeft),
					("Right", totalRight),
					("Status", status),
				]
			else:
				info = [
					("Up", totalUp),
					("Down", totalDown),
					("Status", status),
				]

			# loop over the info tuples and draw them on our frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

			# check to see if we should write the frame to disk
			if writer is not None:
				writer.write(frame)

			# show the output frame
			cv2.imshow("Frame", frame)
			# cv2.imwrite('output/pictests/test{}.jpg'.format(str(temp)), frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

			# increment the total number of frames processed thus far and
			# then update the FPS counter
			totalFrames += 1
			fps.update()

		# stop the timer and display FPS information
		fps.stop()
		print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

		# check to see if we need to release the video writer pointer
		if writer is not None:
			writer.release()

		# if we are not using a video file, stop the camera video stream
		# if not self.args.get("input", False):
		self.vs.stop()

		# otherwise, release the video file pointer
		# else:
		# 	vs.release()

		# close any open windows
		cv2.destroyAllWindows()

if __name__=="__main__":
	# self.args = {"prototxt": "mobilenet_ssd/MobileNetSSD_deploy.prototxt", 
	# 	"model": "mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
	# 	"input": "http://admin:750801@98.199.131.202/videostream.cgi?rate=0"
	# 	"confidence": 0.4
	# 	"skip-frames": 30
	# 	"direction": "updown"
	# }

	try:
		home_counter = PeopleCounter(threadID=1, name="home_cam",prototxt='mobilenet_ssd/MobileNetSSD_deploy.prototxt', 
			model='mobilenet_ssd/MobileNetSSD_deploy.caffemodel', 
			input='http://admin:750801@98.199.131.202/videostream.cgi?rate=0', 
			output=None,
			confidence=0.4, 
			skip_frames=30, 
			direction="rightleft",
			enter_direction="right"
		)
		# beach_counter = PeopleCounter(threadID=2, name="beach_cam",prototxt='mobilenet_ssd/MobileNetSSD_deploy.prototxt', 
		# 	model='mobilenet_ssd/MobileNetSSD_deploy.caffemodel', 
		# 	input='http://213.34.225.97:8080/mjpg/video.mjpg', 
		# 	output=None,
		# 	confidence=0.4, 
		# 	skip_frames=30, 
		# 	direction="rightleft"
		# )

		home_counter.start()
		# beach_counter.start()
	except AttributeError as e:
		print("Video stream is invalid or offline. ")
	except Exception as e:
		print("An unknown error occurred opening the video streams. ")
		print(e)

	while home_counter.is_alive():
		if home_counter.countChanged:
			print(home_counter.count)
			home_counter.countChanged = False
	
	print("Exiting main thread")
	
	# vs = cv2.VideoCapture("asd")
	# if not vs.isOpened():
	# 	print("Video stream is invalid or offline")
		
	
	# try:
	# 	vs = VideoStream("asd").start()
	# except:
	# 	print("Video stream is invalid or offline")