import cv2
import mediapipe as mp
from hand_tracking.src.hand_tracker import HandTracker

RIGHT_HAND = 0
LEFT_HAND = 1


class HandDetector:

    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(mode, max_hands, detection_confidence, track_confidence)

    def draw_hands_on_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand=RIGHT_HAND):
        landmark_list = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)
        hands = result.multi_hand_landmarks
        if hands:
            landmark_list = self.__find_landmarks_for_hand(hands, hand, img)
        return landmark_list

    def __find_landmarks_for_hand(self, hands, hand, img):
        landmark_list = []
        selected_hand = None
        first_hand = hands[0]
        first_landmark_x = first_hand.landmark[0].x
        # TODO implementare in maniera generica il riconoscimento delle tue mani(Magari con face detection)
        if len(hands) == 1:
            if (hand == RIGHT_HAND and first_landmark_x < 0.5) or (hand == LEFT_HAND and first_landmark_x > 0.5):
                selected_hand = first_hand
        else:
            second_hand = hands[1]
            second_landmark_x = second_hand.landmark[0].x
            if hand == RIGHT_HAND:
                if first_landmark_x < second_landmark_x:
                    selected_hand = first_hand
                else:
                    selected_hand = second_hand
            else:
                if first_landmark_x < second_landmark_x:
                    selected_hand = second_hand
                else:
                    selected_hand = first_hand
        if selected_hand is None:
            return landmark_list
        for id, landmark in enumerate(selected_hand.landmark):
            height, width, c = img.shape
            cx, cy, cz = int(landmark.x * width), int(landmark.y * height), landmark.z
            landmark_list.append([id, cx, cy, cz, landmark.x, landmark.y, landmark.z])
        return landmark_list

class HandPalmDetector:

    BASE_PATH = "hand_tracking/models/"
    PALM_MODEL_PATH = BASE_PATH + "palm_detection_without_custom_op.tflite"
    LANDMARK_MODEL_PATH = BASE_PATH + "hand_landmark.tflite"
    ANCHORS_PATH = BASE_PATH + "anchors.csv"

    def __init__(self, connection_color=(255, 0, 0), thickness=2, box_shift=0.2, box_enlarge=1):
        self.connection_color = connection_color
        self.thickness = thickness
        self.detector = HandTracker(
            False,
            self.PALM_MODEL_PATH,
            self.LANDMARK_MODEL_PATH,
            self.ANCHORS_PATH,
            box_shift=box_shift,
            box_enlarge=box_enlarge
        )

    def draw_hands_box_on_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        points, bbox = self.detector(img_rgb)
        if points is not None:
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[1][0]), int(bbox[1][1])
            x3, y3 = int(bbox[2][0]), int(bbox[2][1])
            x4, y4 = int(bbox[3][0]), int(bbox[3][1])

            cv2.line(img_rgb, (x1, y1), (x2, y2), self.connection_color, self.thickness)
            cv2.line(img_rgb, (x2, y2), (x3, y3), self.connection_color, self.thickness)
            cv2.line(img_rgb, (x3, y3), (x4, y4), self.connection_color, self.thickness)
            cv2.line(img_rgb, (x4, y4), (x1, y1), self.connection_color, self.thickness)
        return img_rgb

    def find_cropped_hand_image(self, img, width=200, height=200):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        points, bbox = self.detector(img_rgb)
        cropped_img = []
        if points is not None:
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[1][0]), int(bbox[1][1])
            x3, y3 = int(bbox[2][0]), int(bbox[2][1])
            x4, y4 = int(bbox[3][0]), int(bbox[3][1])

            top_left_x = max([min([x1,x2,x3,x4]), 0])
            top_left_y = max([min([y1,y2,y3,y4]), 0])
            bot_right_x = max([x1,x2,x3,x4,0])
            bot_right_y = max([y1,y2,y3,y4,0])
            dim = (width, height)
            cropped_img = img_rgb[top_left_y:bot_right_y, top_left_x:bot_right_x]
            cropped_img  = cv2.resize(cropped_img, dim, interpolation = cv2.INTER_AREA)
        return cropped_img
















