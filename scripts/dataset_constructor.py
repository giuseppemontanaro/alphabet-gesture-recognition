import pandas as pd
import cv2 as cv
import HandTrackingModule as htm
import time

from threading import Thread
from landmark_operator import get_landmark_distance_row


right_hand = htm.RIGHT_HAND;
left_hand = htm.LEFT_HAND;

wCam, hCam = 640, 320
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

num_landmark = 21
num_labels = 500
DETECTION_CONFIDENCE = 0.80
TRACK_CONFIDENCE = 0.85

detector = htm.HandDetector(detection_confidence=DETECTION_CONFIDENCE, track_confidence=TRACK_CONFIDENCE)



while True:
    label = input("inserisci numero etichetta (lettera qualsiasi per uscire): ")
    if not label.isnumeric():
        break
        
    df = pd.DataFrame()
    while df.shape[0] < num_labels:
        success, img = cap.read()
        img = detector.draw_hands_on_image(img)
        landmark_list_right = detector.find_position(img, right_hand)
        landmark_list_left = detector.find_position(img, left_hand)
        cv.imshow("Img", img)
        cv.waitKey(1)
        if len(landmark_list_right) > 0 and len(landmark_list_left) > 0:
        	continue
        landmark_list = None
        if len(landmark_list_right) > 0:
        	landmark_list = landmark_list_right
        else: 
        	landmark_list = landmark_list_left
        if(len(landmark_list) > 0):
            df = df.append(get_landmark_distance_row(num_landmark, landmark_list), ignore_index=True)
            print(df.shape[0])

    df[num_landmark * 2] = label
    df.to_csv(f'../dataset/{label}.csv', index=False)

cap.release()
