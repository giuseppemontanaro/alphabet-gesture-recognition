import pandas as pd
import cv2
import HandTrackingModule as htm

lettera = input("inserisci numero etichetta: ");
hand = htm.RIGHT_HAND;
wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetector(detection_confidence=0.7)
df = pd.DataFrame()
while df.shape[0] < 50:
    success, img = cap.read()
    landmark_list = detector.find_position(img, hand)
    if len(landmark_list) > 0:
        row = {}
        j = 0;
        for i in range(42):
            index = int(i / 2)
            row[i] = landmark_list[index][1]
            i += 1
            row[i] = landmark_list[index][2]
        df = df.append(row, ignore_index=True)
    print(df.shape[0])
df[42] = lettera;
print(df);
df.to_csv(f'dataset/{lettera}.csv', index=False);

