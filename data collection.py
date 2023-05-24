import cv2
from cvzone.HandTrackingModule import HandDetector

import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

folder = "Sign language data/Z"
counter = 0

while True:
    success,  img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCropp = img[y-offset:y+h+offset, x-offset:x+w+offset]

        cv2.imshow("ImageCropp", imgCropp)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time() }.jpg', imgCropp)
        print(counter)
