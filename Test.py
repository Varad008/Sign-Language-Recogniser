import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
label = ["A", "B", "C"]

folder = "Sign language data/C"
counter = 0

while True:
    success,  img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCropp = img[y-offset:y+h+offset, x-offset:x+w+offset]

        cv2.imshow("ImageCropp", imgCropp)
        prediction, imdex = classifier.getPrediction(imgCropp)
        print(prediction, imdex)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

