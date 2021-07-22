import cv2 as cv
from hand_tracking.src.hand_tracker import HandTracker
from datetime import datetime
import os


def get_orthogonal_rect(image, points):
    min_x = image.shape[1]
    max_x = -1
    min_y = image.shape[0]
    max_y = -1
    for point in points:
        x = point[0]
        y = point[1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    return int(min_x), int(max_x), int(min_y), int(max_y)

BASE_PATH = "hand_tracking/models/"
PALM_MODEL_PATH = BASE_PATH + "palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = BASE_PATH + "hand_landmark.tflite"
ANCHORS_PATH = BASE_PATH + "anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

NUM_LABELS = 500

detector = HandTracker(
    False,
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1
)

capture = cv.VideoCapture(0)
wCam, hCam = 640, 320
capture.set(3, wCam)
capture.set(4, hCam)

while True:
    label = input("inserisci numero etichetta (lettera qualsiasi per uscire): ")
    if not label.isnumeric():
        break

    dir = os.path.join(f"../dataset/imgs/{label}")
    if not os.path.exists(dir):
        os.mkdir(dir)

    num_example = 0
    while num_example < NUM_LABELS:
        hasFrame, frame = capture.read()
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        points, bbox = detector(image)
        if points is not None:
            min_x, max_x, min_y, max_y = get_orthogonal_rect(image, bbox)
            cv.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), THICKNESS)
            cropped = image[min_y : max_y-1, min_x : max_x-1]
            if cropped.size != 0:
                cropped = cv.cvtColor(cropped, cv.COLOR_BGR2RGB)
                now = datetime.now()
                timestamp = datetime.timestamp(now)
                cv.imwrite(f'../dataset/imgs/{label}/{timestamp}.png', cropped)
                num_example += 1
                print(num_example)
        cv.imshow("Img", image)
        key = cv.waitKey(1)
