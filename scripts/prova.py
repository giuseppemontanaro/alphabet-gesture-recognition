import cv2
import argparse
from hand_tracking.src.hand_tracker import HandTracker
from datetime import datetime
import os

WINDOW = "Hand Tracking"
BASE_PATH = "hand_tracking/models/"
PALM_MODEL_PATH = BASE_PATH + "palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = BASE_PATH + "hand_landmark.tflite"
ANCHORS_PATH = BASE_PATH + "anchors.csv"

CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]



detector = HandTracker(
    False,
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1
)

label = input("inserisci numero etichetta (lettera qualsiasi per uscire): ")
dir = os.path.join(f"../dataset/imgs/{label}")
if not os.path.exists(dir):
    os.mkdir(dir)

while hasFrame:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bbox = detector(image)
    cropped_img = frame
    if points is not None:
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[1][0]), int(bbox[1][1])
        x3, y3 = int(bbox[2][0]), int(bbox[2][1])
        x4, y4 = int(bbox[3][0]), int(bbox[3][1])
        #cv2.line(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), CONNECTION_COLOR, THICKNESS)
        #cv2.line(frame, (int(bbox[1][0]), int(bbox[1][1])), (int(bbox[2][0]), int(bbox[2][1])), CONNECTION_COLOR, THICKNESS)
        #cv2.line(frame, (int(bbox[2][0]), int(bbox[2][1])), (int(bbox[3][0]), int(bbox[3][1])), CONNECTION_COLOR, THICKNESS)
        #cv2.line(frame, (int(bbox[3][0]), int(bbox[3][1])), (int(bbox[0][0]), int(bbox[0][1])), CONNECTION_COLOR, THICKNESS)
        #Blu
        #cv2.circle(frame, (int(bbox[0][0]), int(bbox[0][1])), 20, (255, 0, 0), 3)
        #Verde
        #cv2.circle(frame, (int(bbox[1][0]), int(bbox[1][1])), 20, (0, 255, 0), 3)
        #Rosso
        #cv2.circle(frame, (int(bbox[2][0]), int(bbox[2][1])), 20, (0, 0, 255), 3)
        #Viola
        #cv2.circle(frame, (int(bbox[3][0]), int(bbox[3][1])), 20, (0, 255, 255), 3)

        top_left_x = min([x1,x2,x3,x4])
        top_left_y = min([y1,y2,y3,y4])
        bot_right_x = max([x1,x2,x3,x4])
        bot_right_y = max([y1,y2,y3,y4])
        top_left_x = max([0, top_left_x])
        top_left_y = max([0, top_left_y])
        bot_right_x = max([0, bot_right_x])
        bot_right_y = max([0, bot_right_y])
        cropped_img = frame[top_left_y:bot_right_y, top_left_x:bot_right_x]

        width = 200
        height = 200
        dim = (width, height)

        cropped_img  = cv2.resize(cropped_img , dim, interpolation = cv2.INTER_AREA)
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        cv2.imwrite(f'../dataset/imgs/{label}/{timestamp}.png', cropped_img)
        print("saved img")

    cv2.imshow(WINDOW, cropped_img)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()