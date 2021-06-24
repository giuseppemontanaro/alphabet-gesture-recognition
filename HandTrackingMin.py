import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands   # Da chiamare per fare il tracking delle mani
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convertiamo l'immagine per lavorare in RGB
    result = hands.process(imgRGB)
    # print(result.multi_hand_landmarks) Effettua il tracking delle mani e se non le trova fa la detection
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id, landmark in enumerate(hand_landmarks.landmark):
                # print(f'id: {id}, landmark: {landmark}')
                # Abbiamo 21 id a cui sono associati dei valori decimali di x, y, z
                # Per ottenere il valore in pixel vanno moltiplicati per le dimensioni dell'immagini
                height, width, c = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height) # Valori in pixel della x e della y
                print(f'id: {id}, cx: {cx}, cy: {cy}')
                if id == 8:   # Grazie a questo codice posso evidenziare nella figura il punto con un determinato id
                    cv2.circle(img, (cx, cy), 20, (255, 0, 0))
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)


    cv2.imshow("Image", img)
    cv2.waitKey(1)