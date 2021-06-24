import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 1640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
myList.sort()
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)

detector = htm.HandDetector(detection_confidence=0.75)

top_finger_list = [8, 12, 16, 20]

while True:
    success, img = cap.read()
    detector.draw_hands_on_image(img)
    landmarks_list_right = detector.find_position(img, htm.RIGHT_HAND)

    if len(landmarks_list_right) != 0:

        finger_list = []
        if landmarks_list_right[4][1] > landmarks_list_right[3][1]:
            finger_list.append(1)
        else:
            finger_list.append(0)

        for i in range(len(top_finger_list)):
            if landmarks_list_right[top_finger_list[i]][2] < landmarks_list_right[top_finger_list[i] - 2][2]:
                finger_list.append(1)
            else:
                finger_list.append(0)


        num_fingers = finger_list.count(1)
        #print(num_fingers)
        img[0:200, 0:200] = cv2.resize(overlayList[num_fingers - 1], (200, 200))


    cv2.imshow("Image", img)
    cv2.waitKey(1)




