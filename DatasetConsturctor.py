import pandas as pd
import cv2
import HandTrackingModule as htm

label = input("inserisci numero etichetta: ");
right_hand = htm.RIGHT_HAND;
left_hand = htm.LEFT_HAND;

wCam, hCam = 640, 320
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


detector = htm.HandDetector(detection_confidence=0.95, track_confidence=0.8)
num_landmark = 21
num_labels = 2000
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
	    row = {}
	    j = 0;
	    for i in range(num_landmark * 2):
	        index = int(i / 2)
	        row[i] = landmark_list[index][4]
	        i += 1
	        row[i] = landmark_list[index][5]
	    df = df.append(row, ignore_index=True)
	    print(df.shape[0])
cap.release()
df[num_landmark * 2] = label
print(df)
df.to_csv(f'dataset/{label}.csv', index=False)

