# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# import time
#
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
#
# offset = 20
# imgsize = 300
#
# folder = "Data/New"
# counter = 0
#
# while True:
#     success, img = cap.read()
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
#
#         else:
#             k = imgsize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgsize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgsize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, :] = imgResize
#
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)
#     if key == ord("s"):
#         counter += 1
#         cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
#         print(counter)
#


import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import math
import os
import time
from cvzone.HandTrackingModule import HandDetector


class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam with Hand Detection")

        # Initialize HandDetector
        self.detector = HandDetector(maxHands=1)

        # Parameters for image processing
        self.offset = 20
        self.imgsize = 300

        # Create input section
        self.input_label = tk.Label(root, text="Enter directory name:")
        self.input_label.pack()

        self.directory_entry = tk.Entry(root)
        self.directory_entry.pack()

        self.save_button = tk.Button(root, text="Save", command=self.save_image)
        self.save_button.pack()

        # Create counter section
        self.counter_label = tk.Label(root, text="Counter: 0")
        self.counter_label.pack()

        # Create video capture widget
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Start webcam feed
        self.video_capture = cv2.VideoCapture(0)
        self.update_video()

        # Counter
        self.counter = 0

        # Create back button
        self.back_button = tk.Button(root, text="Back", command=self.open_main_py)
        self.back_button.pack()

    def save_image(self):
        directory_name = self.directory_entry.get()
        folder = f"Data/{directory_name}"

        # Check if directory exists, if not, create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        success, img = self.video_capture.read()
        if success:
            # Find hands in the frame
            hands, _ = self.detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                img_crop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

                cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', img_crop)
                self.counter += 1
                self.counter_label.config(text=f"Counter: {self.counter}")

    def update_video(self):
        # Read frame from the webcam
        ret, frame = self.video_capture.read()

        if ret:
            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find hands in the frame
            hands, _ = self.detector.findHands(frame_rgb)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                img_crop = frame_rgb[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

                if y - 20 > 0:
                    cv2.putText(frame_rgb, "Hand Detected", (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

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
    app = WebcamApp(root)
    root.mainloop()
