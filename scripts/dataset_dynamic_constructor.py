import pandas as pd
import cv2 as cv
import HandTrackingModule as htm
import time

from landmark_operator import get_landmark_distance_row
from video_capture_thread import VideoCaptureThread

right_hand = htm.RIGHT_HAND;
left_hand = htm.LEFT_HAND;

wCam, hCam = 640, 320
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(detection_confidence=0.80, track_confidence=0.85)
num_landmark = 21
num_landmark_dist = 32
num_labels = 100

print("calcolando fps...")
start = time.time()
for i in range(0, 30):
    _, frame = cap.read()
end = time.time()
seconds = end - start
fps  = 30 / seconds;
print(f"\nTempo: {seconds} secondi, fps: {fps}")

data = []
while True:
    label = input("\ninserisci numero etichetta (lettera qualsiasi per uscire): ")
    if not label.isnumeric():
        break
    
    img_thread = VideoCaptureThread(fps)
    img_thread.start()
    df_frames = pd.DataFrame()

    while df_frames.shape[0] < num_labels:
        print("inizia tra un secondo...")
        time.sleep(1)
        print("inizia")
        while df_frames.shape[0] < fps:
            img = img_thread.read()
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
                
            if len(landmark_list) > 0:
                df_frames = df_frames.append(get_landmark_distance_row(num_landmark, landmark_list), ignore_index=True)
            
            if df_frames.shape == fps:
                frames = df_frames.to_numpy()
                row = frames.reshape((1, fps * num_landmark_dist))
                data.append(row)
                df_frames = pd.DataFrame()
        print("fine\n")

    img_thread.stop()
    df = pd.DataFrame(data)
    df[df.shape[1] + 1] = label
    df.to_csv(f'../dataset/dynamic/{label}.csv', index=False)
