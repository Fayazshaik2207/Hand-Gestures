# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import time
#
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
#
#
# offset = 20
# imgsize = 300
#
# folder = "Data/C"
# counter = 0
#
# labels = ["A", "B", "C"]
#
# while True:
#     success, img = cap.read()
#     imgOutPut = img.copy()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#
#         imgWhite = np.ones((imgsize, imgsize,3), np.uint8)*255
#
#
#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#
#         imgCropShape = imgCrop.shape
#         aspectRatio = h / w
#
#         if aspectRatio > 1:
#             k = imgsize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgsize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgsize - wCal)/2)
#             imgWhite[:, wGap:wCal+wGap] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             text = labels[index]
#             print(prediction, text)
#
#         else:
#             k = imgsize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgsize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgsize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, :] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             text = labels[index]
#             print(prediction, text)
#
#
#         cv2.putText(imgOutPut,labels[index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
#         cv2.rectangle(imgOutPut,(x- offset,y - offset),(x+w+offset,y+h+offset),(255,0,255),4)
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#
#     cv2.imshow("Image", imgOutPut)
#     cv2.waitKey(1)






# the main  code

# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import time
# import pyttsx3
#
# # Initialize the text-to-speech engine
# engine = pyttsx3.init()
#
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
#
# offset = 20
# imgsize = 300
#
# folder = "Data/C"
# counter = 0
#
# labels = ["Hi, Stop", "Love You", "Victory", "Rock On", "Call Me", "Like", "DisLike", "Raise Hand", "Good Luck", "Agreement", "Protest, Power", "Pinch", "LoveHope", "Greetings", "Question", "Smile"]
#
# while True:
#     success, img = cap.read()
#     imgOutPut = img.copy()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#
#         imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
#
#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#
#         imgCropShape = imgCrop.shape
#         aspectRatio = h / w
#
#         if aspectRatio > 1:
#             k = imgsize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgsize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgsize - wCal) / 2)
#             imgWhite[:, wGap:wCal + wGap] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             text = labels[index]
#             print(prediction, text)
#         else:
#             k = imgsize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgsize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgsize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, :] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             text = labels[index]
#             print(prediction, text)
#
#         # Convert predicted text to voice
#         engine.say(text)
#         engine.runAndWait()
#
#         cv2.putText(imgOutPut, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
#         cv2.rectangle(imgOutPut, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#
#     cv2.imshow("Image", imgOutPut)
#     cv2.waitKey(1)




import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

class SignLanguageRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection with Hand Gestures")

        # Initialize the text-to-speech engine
        self.engine = pyttsx3.init()

        # Initialize HandDetector and Classifier
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

        # Parameters for image processing
        self.offset = 20
        self.imgsize = 300

        # Labels for sign language gestures
        self.labels = ["Agreement","Call me","Dislike","Greetings","Hi","I love You","Like","Love Hope","Protest","Question","Smile"]
        # Create heading label
        self.heading_label = tk.Label(root, text="Sign Language Detection with Hand Gestures", font=("Helvetica", 16))
        self.heading_label.pack(side="top", pady=10)

        # Create video capture widget
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        # Create label for displaying recognized text
        self.recognized_text_label = tk.Label(root, text="Gesture recognised text:", font=("Helvetica", 16))
        self.recognized_text_label.pack()

        # Create label for displaying recognized gesture
        self.gesture_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.gesture_label.pack(pady=5)

        # Create back button
        self.back_button = tk.Button(root, text="Back", command=self.open_main_py)
        self.back_button.pack(pady=10)

        # Start webcam feed
        self.video_capture = cv2.VideoCapture(0)
        self.update_video()

    def update_video(self):
        # Read frame from the webcam
        ret, frame = self.video_capture.read()

        if ret:
            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find hands in the frame
            hands , _ = self.detector.findHands(frame_rgb)

            if hands:
                # Get the first detected hand
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Crop hand region
                img_crop = frame_rgb[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

                # Resize hand region to the desired size
                img_white = np.ones((self.imgsize, self.imgsize, 3), np.uint8) * 255
                img_resize = cv2.resize(img_crop, (self.imgsize, self.imgsize))

                # Update the label with the recognized gesture
                prediction, index = self.classifier.getPrediction(img_resize, draw=False)
                text = self.labels[index]
                self.engine.say(text)
                self.engine.runAndWait()

                # Draw text and rectangle around the hand
                cv2.rectangle(frame_rgb, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset),
                              (255, 0, 255), 4)

                # Update recognized text label
                self.gesture_label.config(text=text)

            # Convert frame to ImageTk format
            frame_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

            # Update video label with the new frame
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk

        # Repeat the update process after a delay
        self.root.after(10, self.update_video)

    def open_main_py(self):
        os.system("python main.py")

# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageRecognitionApp(root)
    root.mainloop()
