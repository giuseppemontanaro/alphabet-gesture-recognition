import cv2 as cv
import numpy as np
import HandTrackingModule as htm

from hand_tracking.src.hand_tracker import HandTracker
from tensorflow.keras.models import load_model
from tensorflow import expand_dims



#MODEL_CNN = load_model('../models/cnn_model_h_q.h5')
#MODEL_CNN = load_model('../models/cnn_model.h5')
#MODEL_CNN = load_model('../models/cnn_model_finale.h5')
# BUONO MODEL_CNN = load_model('../models/cnn_model_non_simmetrico.h5')
MODEL_CNN = load_model('../models/cnn_model_simmetrico_new.h5')

MODEL_HAND_TRACKING_PATH = "hand_tracking/models/"
PALM_MODEL_PATH = MODEL_HAND_TRACKING_PATH + "palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = MODEL_HAND_TRACKING_PATH + "hand_landmark.tflite"
ANCHORS_PATH = MODEL_HAND_TRACKING_PATH + "anchors.csv"



capture = cv.VideoCapture(0)
cont = 0
detector = htm.HandPalmDetector()
while True:
	hasFrame, image = capture.read()
	img = detector.find_cropped_hand_image(image, 150, 150)
	show_img = img
	if len(img) > 0:
		#img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		img_rgb = img
		show_img = img_rgb
		#img_rgb = expand_dims(img_rgb, axis=-1)
		img_rgb = img_rgb.reshape(1, 150, 150, 3)
		#img_rgb = img_rgb /255

		predictions = MODEL_CNN.predict(img_rgb)
		#print(predictions)
		idx = np.argmax(predictions)
		#print(idx)
		if idx == 0: print('p')
		if idx == 1: print('q')
		if idx == 2: print('h')

		cv.imshow("Img", show_img)
	key = cv.waitKey(1)