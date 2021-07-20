import cv2 as cv
from hand_tracking.src.hand_tracker import HandTracker

BASE_PATH = "hand_tracking/models/"
PALM_MODEL_PATH = BASE_PATH + "palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = BASE_PATH + "hand_landmark.tflite"
ANCHORS_PATH = BASE_PATH + "anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

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
    hasFrame, frame = capture.read()
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    points, bbox = detector(image)
    if points is not None:
        cv.line(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), CONNECTION_COLOR, THICKNESS)
        cv.line(frame, (int(bbox[1][0]), int(bbox[1][1])), (int(bbox[2][0]), int(bbox[2][1])), CONNECTION_COLOR, THICKNESS)
        cv.line(frame, (int(bbox[2][0]), int(bbox[2][1])), (int(bbox[3][0]), int(bbox[3][1])), CONNECTION_COLOR, THICKNESS)
        cv.line(frame, (int(bbox[3][0]), int(bbox[3][1])), (int(bbox[0][0]), int(bbox[0][1])), CONNECTION_COLOR, THICKNESS)

    cv.imshow("Img", frame)
    hasFrame, frame = capture.read()
    key = cv.waitKey(1)
    if key == 27:
        break