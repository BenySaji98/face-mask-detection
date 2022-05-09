from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

prototxtPath = None
weightsPath = None
maskNetPath = None
faceNet = None
maskNet = None
vs = None

def load_configs():
    global prototxtPath, weightsPath, maskNetPath, faceNet, maskNet, vs
    """
    start and load the models and cameras
    """

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    if prototxtPath is None:
        raise ValueError("no path to prototxt")
    if weightsPath is None:
        raise ValueError("No path to weights")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(maskNetPath)

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream....",end="")
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2)
    print("started")



def read_camera():
    """
    return frame from camera stream
    """
    if vs is None:
        raise ValueError("Camera is not loaded")

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    return frame


def detect_and_predict_mask(frame, min_confi=0.5):
    """
    Return (faces,preds) after processing the frame
    """
    if faceNet is None:
        raise ValueError('faceNet is not loaded')
    if maskNet is None:
        raise ValueError('maskNet is not loaded')

	# grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize list of faces, locations
	faces,locs,preds = list(),list(),list()


	for i in range(0, detections.shape[2]):

        # filter out weak detections
		confidence = detections[0, 0, i, 2]
        if confidence < min_confi:
            continue

		# compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# ensure the bounding boxes fall within the dimensions of the frame
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# extract the face ROI, convert it from BGR to RGB channel ordering,
        # resize it to 224x224, and preprocess it
		face = frame[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)

		# update list
        faces.append(face)
		locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

def find_mask(confidence=0.5,draw=True):
    """
    return the count of faces with mask and without mask along with the frame.
    frame extraced from camera
    """
    yes_mask, no_mask = 0,0

    frame = read_camera()

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame,confidence)

    # loop over the detected face locations and their corresponding
    # locations

    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text

        if mask > withoutMask:
            yes_mask += 1
            olor = (0, 255, 0)
        else:
            no_mask += 1
            color = (0, 0, 255)


        if draw:
            cv2.putText(frame, label, (startX-50, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
		    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    return yes_mask, no_mask, frame


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,default="mask_detector.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())


    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
    maskNetPath = args['model']

    load_configs()


    try:
        print("use CTRL-C for Keyboard Interrupt")

        # loop over the frames from the video stream
        while True:
            yes_mask, no_mask, frame = find_mask(args['confidence'],True)

            # show the output frame
        	cv2.imshow("Face Mask Detector", frame)

            total_faces = yes_mask + no_mask
            print(f"found {total_faces} faces with {yes_mask} mask")

            if total_faces == 0: # skip
                continue
            if total_faces > 1:
                print("Warning! Only one person allowed!")
                time.sleep(5)
                continue

            if yes_mask == 1:
                print("Found mask")
            else:
                print("No mask found")
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt")
    finally:
        cv2.destroyAllWindows()
        vs.stop()
        print("feed ended")
