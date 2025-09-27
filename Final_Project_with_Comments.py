# Import required libraries
import cv2                                  # OpenCV for image/video capture and processing
from cvzone.HandTrackingModule import HandDetector   # For detecting hand landmarks
from cvzone.ClassificationModule import Classifier   # For classification of gestures
import numpy as np                          # For numerical operations
import math                                 # For resizing calculations
import os                                   # For file/folder handling
import pygame                               # For playing audio
import time                                 # For time-based debounce
from gtts import gTTS                       # Google Text-to-Speech for audio generation

# ===============================
# Configuration
# ===============================
OFFSET = 20            # Extra padding around detected hand when cropping
IMG_SIZE = 300         # Input image size for classifier
LABELS = ["A", "B", "C", "GOOD", "HELLO", "THANK YOU", "YES"]  # Labels to classify
MIN_DELAY = 2          # Delay (in seconds) between voice outputs to avoid repetition
SENTENCE_MAX = 5       # Maximum number of words to display in the sentence

# ===============================
# Setup
# ===============================
cap = cv2.VideoCapture(0)   # Open default camera (webcam index 0)
detector = HandDetector(maxHands=1)   # Detect only one hand at a time
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")   # Load trained model and labels
pygame.mixer.init()   # Initialize pygame for audio playback

# Variables for controlling speech and sentence construction
last_label = None        # Stores last spoken label to avoid repetition
last_spoken_time = 0     # Stores timestamp of last spoken word
sentence = []            # Stores sentence as list of recognized labels

# ===============================
# Pre-generate audio for labels
# ===============================
AUDIO_DIR = "audio_cache"   # Folder for cached audio files
os.makedirs(AUDIO_DIR, exist_ok=True)   # Create folder if it doesn’t exist

def generate_audio_files():
    """Generate mp3 files for all labels once and reuse them."""
    for label in LABELS:  # Loop through each label
        path = os.path.join(AUDIO_DIR, f"{label}.mp3")   # Path to mp3 file
        if not os.path.exists(path):   # If file doesn’t exist, create it
            try:
                tts = gTTS(text=label, lang="en")   # Convert label text to speech
                tts.save(path)                      # Save as mp3
            except Exception as e:                  # Handle errors
                print(f"Error generating audio for {label}: {e}")

generate_audio_files()   # Generate all audio files before loop starts

# ===============================
# Speech function (non-blocking)
# ===============================
def speak(label):
    """Play cached audio file for a label without blocking video loop."""
    try:
        path = os.path.join(AUDIO_DIR, f"{label}.mp3")   # Path to cached audio
        if os.path.exists(path):                         # If audio exists
            pygame.mixer.music.load(path)                # Load the mp3
            pygame.mixer.music.play()                    # Play asynchronously
    except Exception as e:                               # Error handling
        print("Speech Error:", e)

# ===============================
# Safe crop helper
# ===============================
def safe_crop(img, x, y, w, h, offset):
    """Crop with bounds checking so it never crashes."""
    H, W, _ = img.shape   # Get image dimensions
    x1, y1 = max(0, x - offset), max(0, y - offset)   # Top-left (with safety check)
    x2, y2 = min(W, x + w + offset), min(H, y + h + offset)   # Bottom-right
    return img[y1:y2, x1:x2]   # Return cropped region

# ===============================
# Main Loop
# ===============================
while True:
    success, img = cap.read()   # Capture frame from camera
    if not success:             # If no frame, continue loop
        continue

    imgOutput = img.copy()   # Copy of frame for display
    hands, _ = detector.findHands(img)   # Detect hands in frame

    # Draw header bar at top
    cv2.rectangle(imgOutput, (0, 0), (640, 40), (50, 50, 50), -1)
    cv2.putText(imgOutput, "Sign Language Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if hands:   # If hand detected
        hand = hands[0]          # Take first detected hand
        x, y, w, h = hand['bbox']   # Bounding box of hand

        # White background image for classifier
        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        imgCrop = safe_crop(img, x, y, w, h, OFFSET)   # Crop hand with padding

        # Resize hand crop while maintaining aspect ratio
        if w > 0 and h > 0:
            aspectRatio = h / w
            if aspectRatio > 1:   # Tall hand
                k = IMG_SIZE / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                wGap = (IMG_SIZE - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:   # Wide hand
                k = IMG_SIZE / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                hGap = (IMG_SIZE - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Predict gesture using classifier
            prediction, index = classifier.getPrediction(imgWhite)
            confidence = prediction[index]   # Prediction confidence
            label = LABELS[index]            # Predicted label

            # Speak word if not repeated and after delay
            if label != last_label and time.time() - last_spoken_time > MIN_DELAY:
                speak(label)                  # Play audio
                last_label = label            # Update last label
                last_spoken_time = time.time()  # Update timestamp
                sentence.append(label)        # Add word to sentence
                sentence = sentence[-SENTENCE_MAX:]  # Keep only last N words

            # Show prediction text
            cv2.putText(imgOutput, f'{label}: {int(confidence * 100)}%', (x, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw bounding box around hand
            cv2.rectangle(imgOutput, (x - OFFSET, y - OFFSET),
                          (x + w + OFFSET, y + h + OFFSET), (255, 0, 255), 4)

            # Confidence bar above hand
            bar_x, bar_y = x, y - 30
            cv2.rectangle(imgOutput, (bar_x, bar_y), (bar_x + 100, bar_y + 10), (200, 200, 200), 2)
            cv2.rectangle(imgOutput, (bar_x, bar_y),
                          (bar_x + int(confidence * 100), bar_y + 10), (0, 255, 0), -1)

            # Debugging windows
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)

    # Display sentence at bottom
    display_text = ' '.join(sentence)
    cv2.putText(imgOutput, f'Sentence: {display_text}', (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Show final camera output
    cv2.imshow("Image", imgOutput)

    # Key controls
    key = cv2.waitKey(1)
    if key == ord('s'):  # If 's' pressed → save current sentence
        with open("log.txt", "a") as f:
            f.write(display_text + "\n")
        print("Sentence saved to log.txt.")
    elif key == ord('q'):  # If 'q' pressed → quit loop
        break

# ===============================
# Cleanup
# ===============================
cap.release()           # Release camera
cv2.destroyAllWindows() # Close all OpenCV windows
pygame.mixer.quit()     # Quit pygame
