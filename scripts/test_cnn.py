import cv2 as cv
import numpy as np

from hand_tracking.src.hand_tracker import HandTracker
from tensorflow.keras.models import load_model


def get_orthogonal_rect(image, points):
    min_x = image.shape[1]
    max_x = -1
    min_y = image.shape[0]
    max_y = -1
    for point in points:
        x = point[0]
        y = point[1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    return int(min_x), int(max_x), int(min_y), int(max_y)


#MODEL_CNN = load_model('../models/cnn_model_h_q.h5')
#MODEL_CNN = load_model('../models/cnn_model.h5')
MODEL_CNN = load_model('../models/cnn_model_simone.h5')

MODEL_HAND_TRACKING_PATH = "hand_tracking/models/"
PALM_MODEL_PATH = MODEL_HAND_TRACKING_PATH + "palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = MODEL_HAND_TRACKING_PATH + "hand_landmark.tflite"
ANCHORS_PATH = MODEL_HAND_TRACKING_PATH + "anchors.csv"

detector = HandTracker(
    False,
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1
)


capture = cv.VideoCapture(0)
wCam, hCam = 640, 320
capture.set(3, wCam)
capture.set(4, hCam)

while True:
	hasFrame, image = capture.read()
	points, bbox = detector(image)
	if points is not None:
		min_x, max_x, min_y, max_y = get_orthogonal_rect(image, bbox)
		cv.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
		cropped = image[min_y+1 : max_y-1, min_x+1 : max_x-1]
		if cropped.size != 0:
			cropped = cv.resize(cropped, (150, 150))
			cropped = np.expand_dims(cropped, axis=0)
			predictions = MODEL_CNN.predict(cropped)
			idx = np.argmax(predictions)
			print(predictions)
			if idx == 0: print('h')
			if idx == 1: print('p')
			if idx == 2: print('q')
	cv.imshow("Img", image)
	key = cv.waitKey(1)
