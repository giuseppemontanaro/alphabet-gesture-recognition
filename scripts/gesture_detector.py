import cv2
import HandTrackingModule as htm
import pickle
import json
import pandas as pd
from collections import Counter
from os import system


from landmark_operator import get_landmark_distance_row


MODEL = pickle.load(open('../models/model.sav', 'rb'))
ALPHABET = json.load(open('../json/alphabet_mapping.json', 'rb'))
NUM_LANDMARK = 21
NUM_PREDICTION_FRAME = 10


def main():
	wCam, hCam = 640, 320
	cap = cv2.VideoCapture(0)
	cap.set(3, wCam)
	cap.set(4, hCam)
	detector = htm.HandDetector(detection_confidence=0.80, track_confidence=0.85)
	detect(detector, cap)


def detect(detector, cap):
	buffer_letter = []
	word_to_say = ''
	while True:
		success, img = cap.read()
		img = detector.draw_hands_on_image(img)
		cv2.imshow("Img", img)
		cv2.waitKey(1)
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
				if(value > NUM_PREDICTION_FRAME*0.7):
					if key == 16:
						#word_to_say = word_to_say + ' '
						system(f'say {word_to_say}')
						word_to_say = ''
					else:
						word_to_say = word_to_say + ALPHABET[str(key)] 
					print(word_to_say)
				buffer_letter=[]


def predict():
	result = model.predict()


if __name__ == '__main__':
	main()