import pandas as pd
import cv2 as cv
import HandTrackingModule as htm
import time
from pathlib import Path

from threading import Thread
from landmark_operator import get_landmark_distance_row


cap = cv.VideoCapture(0)

num_labels = 500
detector = htm.HandPalmDetector()
img_saved = 0
while True:
    label = input("inserisci numero etichetta (lettera qualsiasi per uscire): ")
    if not label.isnumeric():
        break
        
    while img_saved < num_labels:
        success, img = cap.read()
        #img = detector.draw_hands_on_image(img)
        img = detector.find_cropped_hand_image(img)
        if len(img) > 0:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            cv.imshow("Img", img_rgb)
        cv.waitKey(1)

    #Path('../dataset').mkdir(parents=True, exist_ok=True)
    #df.to_csv(f'../dataset/csv/{label}.csv', index=False)

cap.release()
