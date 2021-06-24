import cv2
import mediapipe as mp

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
            landmark_list.append([id, cx, cy, cz])
        return landmark_list


