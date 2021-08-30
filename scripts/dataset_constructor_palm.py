import pandas as pd
import cv2 as cv
import HandTrackingModule as htm
import time
import os

from pathlib import Path
from threading import Thread
from landmark_operator import get_landmark_distance_row
from datetime import datetime


cap = cv.VideoCapture(0)

num_labels = 500
detector = htm.HandPalmDetector()
img_saved = 0


label = input("inserisci numero etichetta:")
dir = os.path.join(f"../dataset/imgs/{label}")
if not os.path.exists(dir):
    os.mkdir(dir)
    
while img_saved < num_labels:
    success, img = cap.read()
    img_hands = detector.draw_hands_box_on_image(img)
    img = detector.find_cropped_hand_image(img)
    if len(img) > 0:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imshow("Img", img_rgb)
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        cv.imwrite(f'../dataset/imgs/{label}/{timestamp}.png', img_rgb)
        img_saved += 1
        print(f'Saved {img_saved} imgs')
    if len(img_hands) > 0:
        img_hands_rgb = cv.cvtColor(img_hands, cv.COLOR_BGR2RGB)
        cv.imshow("Img_HANDS", img_hands_rgb)
    cv.waitKey(1)




cap.release()
