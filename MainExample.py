import cv2
import HandTrackingModule as htm


def main():
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector()
    while True:
        success, img = cap.read()
        img = detector.draw_hands_on_image(img)
        landmark_list = detector.find_position(img)
        if len(landmark_list) != 0:
            print(landmark_list[4])
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
