import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from gtts import gTTS
import tempfile
import uuid
import os
import pygame
import time

offset = 20
imgSize = 300
labels = ["A", "B", "C","GOOD","HELLO","THANK YOU","YES"]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

pygame.mixer.init()

last_label = None 

def speak(text):
    try:
        tmp_file = tempfile.gettempdir() + f"\\{uuid.uuid4()}.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(tmp_file)

        pygame.mixer.music.load(tmp_file)
        pygame.mixer.music.play()

    
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        os.remove(tmp_file)
    except Exception as e:
        print("Speech Error:", e)

while True:
    success, img = cap.read()
    if not success:
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite)
        print(prediction, index)

        if labels[index] != last_label:
            speak(labels[index])
            last_label = labels[index]

        cv2.putText(imgOutput, labels[index], (x, y - 20),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
