import cv2
import HandTrackingModule as htm
import pickle
import json
import pandas as pd

from landmark_operator import get_landmark_distance_row


MODEL = pickle.load(open('../models/model.sav', 'rb'))
ALPHABET = json.load(open('../json/alphabet_mapping.json', 'rb'))
NUM_LANDMARK = 21


def main():
	wCam, hCam = 640, 320
	cap = cv2.VideoCapture(0)
	cap.set(3, wCam)
	cap.set(4, hCam)
	detector = htm.HandDetector(detection_confidence=0.80, track_confidence=0.85)
	detect(detector, cap)


def detect(detector, cap):
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
			if pred_prob > 0.7:
				prediction = MODEL.predict(model_input)[0]
				print(f'pred: {ALPHABET[str(prediction)]}, with probability: {pred_prob}')


def predict():
	result = model.predict()


if __name__ == '__main__':
	main()