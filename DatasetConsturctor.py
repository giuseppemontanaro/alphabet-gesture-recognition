import pandas as pd
import cv2
import HandTrackingModule as htm
import math

def get_landmark_x_y_row(num_landmark, landmark_list):
    row = {}
    j = 0;
    for i in range(num_landmark * 2):
        index = int(i / 2)
        row[i] = landmark_list[index][4]
        i += 1
        row[i] = landmark_list[index][5]
    return row

def calculate_distance(x1, y1, x2, y2):
	return math.dist([x1, y1], [x2, y2])

def get_landmark_distance_row(num_landmark, landmark_list):
    row = {}
    row_index = 0;
    for i in range(1, num_landmark):
        row[row_index] = calculate_distance(landmark_list[0][4], landmark_list[0][5], landmark_list[i][4], landmark_list[i][5])
        row_index += 1  
    i = 2
    while i < num_landmark - 4:
    	row[row_index] = calculate_distance(landmark_list[i][4], landmark_list[i][5], landmark_list[i + 4][4], landmark_list[i + 4][5])
    	row_index += 1 
    	i += 4
    i = 3
    while i < num_landmark - 4:
    	row[row_index] = calculate_distance(landmark_list[i][4], landmark_list[i][5], landmark_list[i + 4][4], landmark_list[i + 4][5])
    	row_index += 1 
    	i += 4
    i = 4
    while i < num_landmark - 4:
    	row[row_index] = calculate_distance(landmark_list[i][4], landmark_list[i][5], landmark_list[i + 4][4], landmark_list[i + 4][5])
    	row_index += 1 
    	i += 4
    return row

label = input("inserisci numero etichetta: ");
right_hand = htm.RIGHT_HAND;
left_hand = htm.LEFT_HAND;

wCam, hCam = 640, 320
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


detector = htm.HandDetector(detection_confidence=0.95, track_confidence=0.8)
num_landmark = 21
num_labels = 500
df = pd.DataFrame()

while df.shape[0] < num_labels:
    success, img = cap.read()
    img = detector.draw_hands_on_image(img)
    landmark_list_right = detector.find_position(img, right_hand)
    landmark_list_left = detector.find_position(img, left_hand)
    cv2.imshow("Img", img)
    cv2.waitKey(1)
    if len(landmark_list_right) > 0 and len(landmark_list_left) > 0:
    	continue
    landmark_list = None
    if len(landmark_list_right) > 0:
    	landmark_list = landmark_list_right
    else: 
    	landmark_list = landmark_list_left
    if(len(landmark_list) > 0):
        #df = df.append(get_landmark_x_y_row(num_landmark, landmark_list), ignore_index=True)
        df = df.append(get_landmark_distance_row(num_landmark, landmark_list), ignore_index=True)
        print(df.shape[0])

cap.release()
df[num_landmark * 2] = label
print(df)
df.to_csv(f'dataset/{label}.csv', index=False)

