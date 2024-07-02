# import tkinter as tk
# import cv2
# from PIL import Image, ImageTk
# import numpy as np
# import math
# import time
# import pyttsx3
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
#
#
# class SignLanguageRecognitionApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Sign Language Recognition")
#
#         # Initialize the text-to-speech engine
#         self.engine = pyttsx3.init()
#
#         # Initialize HandDetector and Classifier
#         self.detector = HandDetector(maxHands=1)
#         self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
#
#         # Parameters for image processing
#         self.offset = 20
#         self.imgsize = 300
#
#         # Labels for sign language gestures
#         self.labels = ["Hi, Stop", "Love You", "Victory", "Rock On", "Call Me", "Like", "DisLike", "Raise Hand",
#                        "Good Luck", "Agreement", "Protest, Power", "Pinch", "LoveHope", "Greetings", "Question",
#                        "Smile"]
#
#         # Create video capture widget
#         self.video_label = tk.Label(root)
#         self.video_label.pack(pady=10)
#
#         # Start webcam feed
#         self.video_capture = cv2.VideoCapture(0)
#         self.update_video()
#
#     def update_video(self):
#         # Read frame from the webcam
#         ret, frame = self.video_capture.read()
#
#         if ret:
#             # Convert frame to RGB format
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # Find hands in the frame
#             hands, _ = self.detector.findHands(frame_rgb)
#
#             if hands:
#                 # Get the first detected hand
#                 hand = hands[0]
#                 x, y, w, h = hand['bbox']
#
#                 # Crop hand region
#                 img_crop = frame_rgb[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
#
#                 # Resize hand region to the desired size
#                 img_white = np.ones((self.imgsize, self.imgsize, 3), np.uint8) * 255
#                 img_resize = cv2.resize(img_crop, (self.imgsize, self.imgsize))
#
#                 # Update the label with the recognized gesture
#                 prediction, index = self.classifier.getPrediction(img_resize, draw=False)
#                 text = self.labels[index]
#                 self.engine.say(text)
#                 self.engine.runAndWait()
#
#                 # Draw text and rectangle around the hand
#                 cv2.putText(frame_rgb, text, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
#                 cv2.rectangle(frame_rgb, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset),
#                               (255, 0, 255), 4)
#
#             # Convert frame to ImageTk format
#             frame_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
#
#             # Update video label with the new frame
#             self.video_label.configure(image=frame_tk)
#             self.video_label.image = frame_tk
#
#         # Repeat the update process after a delay
#         self.root.after(10, self.update_video)
#
#
# # Main
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = SignLanguageRecognitionApp(root)
#     root.mainloop()


import tkinter as tk
import os


class MainInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection with Hand gestures using Deep Learning")

        # Create main heading
        self.heading_label = tk.Label(root, text="Sign Language Detection with Hand gestures using Deep Learning",
                                      font=("Helvetica", 16))
        self.heading_label.pack(pady=20)

        # Create buttons
        self.data_collection_button = tk.Button(root, text="Data Collection", command=self.open_data_collection)
        self.data_collection_button.pack(pady=10)

        self.test_button = tk.Button(root, text="Test", command=self.open_test)
        self.test_button.pack(pady=10)

    def open_data_collection(self):
        self.root.withdraw()  # Hide main window
        os.system("python DataCollection.py")  # Execute data collection file

    def open_test(self):
        self.root.withdraw()  # Hide main window
        os.system("python Test.py")  # Execute test file


# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = MainInterface(root)
    root.mainloop()
