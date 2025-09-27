import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import time

# Settings
offset = 20
imgSize = 300
labels = ["A", "B", "C"]
confidence_threshold = 0.85  # Ignore low-confidence predictions
cooldown_time = 1.5  # seconds

# Initialize modules
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# State tracking
last_spoken = None
last_spoken_time = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Handle image cropping with boundary checks
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        # Prepare white background
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

        if confidence > confidence_threshold:
            label = labels[index]
            cv2.putText(imgOutput, f'{label} ({int(confidence*100)}%)', (x, y - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

            # Draw rectangle
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (0, 255, 0), 3)

            # Speak if cooldown passed
            current_time = time.time()
            if label != last_spoken or (current_time - last_spoken_time > cooldown_time):
                tts_engine.say(label)
                tts_engine.runAndWait()
                last_spoken = label
                last_spoken_time = current_time

        else:
            cv2.putText(imgOutput, "Low confidence", (x, y - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import pyttsx3
# import time

# # Settings
# offset = 20
# imgSize = 300
# labels = ["A", "B", "C"]
# confidence_threshold = 0.85
# cooldown_time = 2  # seconds

# # Initialize modules
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# tts_engine = pyttsx3.init()
# tts_engine.setProperty('rate', 150)

# # State variables
# last_label = None
# last_time = 0

# while True:
#     success, img = cap.read()
#     if not success:
#         continue

#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)

#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']

#         y1 = max(0, y - offset)
#         y2 = min(img.shape[0], y + h + offset)
#         x1 = max(0, x - offset)
#         x2 = min(img.shape[1], x + w + offset)
#         imgCrop = img[y1:y2, x1:x2]

#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
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

#         prediction, index = classifier.getPrediction(imgWhite, draw=False)
#         confidence = prediction[index]
#         label = labels[index]

#         current_time = time.time()
#         should_speak = False

#         if confidence > confidence_threshold:
#             # Speak if:
#             # 1. Label changed
#             # 2. Enough time passed (cooldown)
#             if label != last_label or (current_time - last_time) > cooldown_time:
#                 should_speak = True

#             if should_speak:
#                 print(f"Speaking: {label}")
#                 tts_engine.say(label)
#                 tts_engine.runAndWait()
#                 last_label = label
#                 last_time = current_time

#             # Show label
#             cv2.putText(imgOutput, f'{label} ({int(confidence*100)}%)', (x, y - 20),
#                         cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
#         else:
#             cv2.putText(imgOutput, "Low confidence", (x, y - 20),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#         cv2.rectangle(imgOutput, (x - offset, y - offset),
#                       (x + w + offset, y + h + offset), (0, 255, 0), 3)

#         cv2.imshow("Image Crop", imgCrop)
#         cv2.imshow("Image White", imgWhite)

#     cv2.imshow("Image", imgOutput)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
