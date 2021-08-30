import cv2
import HandTrackingModule as htm
import pickle
import json
import pandas as pd
import numpy as np
from collections import Counter
from os import system
from tensorflow.keras.models import load_model


from landmark_operator import get_landmark_distance_row


MODEL = pickle.load(open('../models/model.sav', 'rb'))
MODEL_CNN = load_model('../models/cnn_model_simmetrico_new.h5')
ALPHABET = json.load(open('../json/alphabet_mapping.json', 'rb'))
NUM_LANDMARK = 21
NUM_PREDICTION_FRAME = 12
FONT = cv2.FONT_HERSHEY_SIMPLEX
PRECISION_INDEX = 0.7
FALLBACK_INDEX = 2.5


def main():
	wCam, hCam = 640, 320
	cap = cv2.VideoCapture(0)
	cap.set(3, wCam)
	cap.set(4, hCam)
	detector = htm.HandDetector(detection_confidence=0.80, track_confidence=0.85)
	palm_detector = htm.HandPalmDetector()
	detect(detector, palm_detector, cap)


def detect(detector, palm_detector, cap):
	buffer_letter = []
	buffer_letter_cnn = []
	word_to_say = ''
	while True:
		success, img = cap.read()
		img = detector.draw_hands_on_image(img)
		landmark_list = detector.find_position(img, htm.RIGHT_HAND)
		if len(landmark_list) != 0:
			data = get_landmark_distance_row(NUM_LANDMARK, landmark_list)
			model_input = pd.DataFrame(data, index=[0])
			probabilities = MODEL.predict_proba(model_input)[0]
			pred_prob = max(probabilities)
			
			if pred_prob > 0.6:
				prediction = MODEL.predict(model_input)[0]
				buffer_letter.append(prediction)
				print(f'pred: {ALPHABET[str(prediction)]}, with probability: {pred_prob}')
			if len(buffer_letter) == NUM_PREDICTION_FRAME:
				counter = Counter(buffer_letter).most_common(1) 
				dict_counter = dict(counter)
				value = list(dict_counter.values())[0]
				key = list(dict_counter.keys())[0]
				if(value > NUM_PREDICTION_FRAME * PRECISION_INDEX):
					buffer_letter_cnn = []
					if key == 24:
						if word_to_say != '':
							system(f'say {word_to_say}')
							word_to_say = ''
					else:
						word_to_say = word_to_say + ALPHABET[str(key)] 
					print(word_to_say)
				buffer_letter=[]


		img_palm = palm_detector.find_cropped_hand_image(img, 150, 150)
		if len(img_palm) > 0:
			img_palm = img_palm.reshape(1, 150, 150, 3)
			predictions = MODEL_CNN.predict(img_palm)
			idx = np.argmax(predictions)
			buffer_letter_cnn.append(idx)
			if len(buffer_letter_cnn) >= NUM_PREDICTION_FRAME*FALLBACK_INDEX:
				counter = Counter(buffer_letter_cnn).most_common(1) 
				dict_counter = dict(counter)
				value = list(dict_counter.values())[0]
				key = list(dict_counter.keys())[0]
				if(value > NUM_PREDICTION_FRAME * PRECISION_INDEX * FALLBACK_INDEX):
					buffer_letter = []
					if key == 0: 
						word_to_say = word_to_say + 'p'
					if key == 1: 
						word_to_say = word_to_say + 'q'
					if key == 2: 
						word_to_say = word_to_say + 'h'
				print(f'reset with {dict_counter}')
				buffer_letter_cnn = []
		cv2.putText(img, word_to_say,(100,50), FONT, 1.8,(0,0,255),2)
		cv2.imshow("Img", img)
		cv2.waitKey(1)


def predict():
	result = model.predict()


if __name__ == '__main__':
	main()