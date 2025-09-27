# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math

# offset = 20
# imgSize = 300
# labels = ["A", "B", "C"]

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# while True:
#     success, img = cap.read()
#     if not success:
#         continue

#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)

#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']

#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

#         aspectRatio = h / w

#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap:wGap + wCal] = imgResize
#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap:hGap + hCal, :] = imgResize

#         prediction, index = classifier.getPrediction(imgWhite)
#         print(prediction, index)

#         cv2.putText(imgOutput, labels[index], (x, y - 20),
#                     cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
#         cv2.rectangle(imgOutput, (x - offset, y - offset),
#                       (x + w + offset, y + h + offset), (255, 0, 255), 4)

#         cv2.imshow("Image Crop", imgCrop)
#         cv2.imshow("Image White", imgWhite)

#     cv2.imshow("Image", imgOutput)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from gtts import gTTS
from playsound import playsound
import os

# Settings
offset = 20
imgSize = 300
labels = ["A", "B", "C"]
confidence_threshold = 0.85
cooldown_time = 2  # seconds
audio_file = "speak.mp3"

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Track last spoken letter and time
last_label = None
last_time = 0

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)
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

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
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

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        confidence = prediction[index]
        label = labels[index]

        current_time = time.time()
        should_speak = False

        if confidence > confidence_threshold:
            # Speak if it's a new letter or cooldown passed
            if label != last_label or (current_time - last_time) > cooldown_time:
                should_speak = True

            if should_speak:
                print(f"Speaking: {label}")
                speak(label)
                last_label = label
                last_time = current_time

            cv2.putText(imgOutput, f'{label} ({int(confidence*100)}%)', (x, y - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        else:
            cv2.putText(imgOutput, "Low confidence", (x, y - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (0, 255, 0), 3)

        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
