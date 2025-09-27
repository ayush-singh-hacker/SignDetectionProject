import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import pygame
import time
from gtts import gTTS

# ===============================
# Configuration
# ===============================
OFFSET = 20
IMG_SIZE = 300
LABELS = ["A", "B", "C", "GOOD", "HELLO", "THANK YOU", "YES"]
MIN_DELAY = 2   # seconds between voice outputs
SENTENCE_MAX = 5  # how many words to keep in sentence

# ===============================
# Setup
# ===============================
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
pygame.mixer.init()

last_label = None
last_spoken_time = 0
sentence = []

# ===============================
# Pre-generate audio for labels
# ===============================
AUDIO_DIR = "audio_cache"
os.makedirs(AUDIO_DIR, exist_ok=True)

def generate_audio_files():
    """Generate mp3 files for all labels once and reuse them."""
    for label in LABELS:
        path = os.path.join(AUDIO_DIR, f"{label}.mp3")
        if not os.path.exists(path):
            try:
                tts = gTTS(text=label, lang="en")
                tts.save(path)
            except Exception as e:
                print(f"Error generating audio for {label}: {e}")

generate_audio_files()

# ===============================
# Speech function (non-blocking)
# ===============================
def speak(label):
    """Play cached audio file for a label without blocking video loop."""
    try:
        path = os.path.join(AUDIO_DIR, f"{label}.mp3")
        if os.path.exists(path):
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
    except Exception as e:
        print("Speech Error:", e)

# ===============================
# Safe crop helper
# ===============================
def safe_crop(img, x, y, w, h, offset):
    """Crop with bounds checking so it never crashes."""
    H, W, _ = img.shape
    x1, y1 = max(0, x - offset), max(0, y - offset)
    x2, y2 = min(W, x + w + offset), min(H, y + h + offset)
    return img[y1:y2, x1:x2]

# ===============================
# Main Loop
# ===============================
while True:
    success, img = cap.read()
    if not success:
        continue

    imgOutput = img.copy()
    hands, _ = detector.findHands(img)

    # Header bar
    cv2.rectangle(imgOutput, (0, 0), (640, 40), (50, 50, 50), -1)
    cv2.putText(imgOutput, "Sign Language Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        imgCrop = safe_crop(img, x, y, w, h, OFFSET)

        # Resize with aspect ratio preserved
        if w > 0 and h > 0:
            aspectRatio = h / w
            if aspectRatio > 1:  # Tall hand
                k = IMG_SIZE / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                wGap = (IMG_SIZE - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:  # Wide hand
                k = IMG_SIZE / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                hGap = (IMG_SIZE - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Prediction
            prediction, index = classifier.getPrediction(imgWhite)
            confidence = prediction[index]
            label = LABELS[index]

            # Speak (debounce)
            if label != last_label and time.time() - last_spoken_time > MIN_DELAY:
                speak(label)
                last_label = label
                last_spoken_time = time.time()

                sentence.append(label)
                sentence = sentence[-SENTENCE_MAX:]  # keep only last N words

            # Visual Feedback
            cv2.putText(imgOutput, f'{label}: {int(confidence * 100)}%', (x, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.rectangle(imgOutput, (x - OFFSET, y - OFFSET),
                          (x + w + OFFSET, y + h + OFFSET), (255, 0, 255), 4)

            # Confidence bar
            bar_x, bar_y = x, y - 30
            cv2.rectangle(imgOutput, (bar_x, bar_y), (bar_x + 100, bar_y + 10), (200, 200, 200), 2)
            cv2.rectangle(imgOutput, (bar_x, bar_y),
                          (bar_x + int(confidence * 100), bar_y + 10), (0, 255, 0), -1)

            # Debug windows
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)

    # Show sentence
    display_text = ' '.join(sentence)
    cv2.putText(imgOutput, f'Sentence: {display_text}', (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Show final output
    cv2.imshow("Image", imgOutput)

    # Key controls
    key = cv2.waitKey(1)
    if key == ord('s'):  # Save sentence
        with open("log.txt", "a") as f:
            f.write(display_text + "\n")
        print("Sentence saved to log.txt.")
    elif key == ord('q'):  # Quit
        break

# ===============================
# Cleanup
# ===============================
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
