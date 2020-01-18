import numpy as np
import cv2
import argparse
import imutils
from threading import Thread



# Class to read and process IP camera frames
class ipstream:

    # Initialize IP camera stream
    def initializeStream(self):
        self.image_name = "frame"
        self.stream = 'rtsp://admin:Vother420@169.254.70.20' + ':554/ch1/main/av_stream'
        #'rtsp://admin:Vother420@169.254.70.20:554/ch1/main/av_stream'
        #use your own ip address's rstp address, username & password
        #this is the rstp address of a hik vision camera
        self.capture = cv2.VideoCapture(self.stream)
        (self.status, self.raw_frame) = self.capture.read()
        self.frame = self.raw_frame
        self.stopped = False
        self.image_width = 900
        self.image_height = 500

    def start(self):
        Thread(target=self.captureFrames, args=()).start()
        return self

    # Check if stream is online
    def isOpened(self):
        return self.capture.isOpened()
   
    # Return most recent frame from stream
    def getFrame(self):
        return self.frame

    # Constantly capture frames
    def captureFrames(self):
        while True:
            if self.stopped:
                return
            (self.status, self.raw_frame) = self.capture.read()
            while not self.status:
                (self.status, self.raw_frame) = self.capture.read()
            self.frame = imutils.resize(self.raw_frame, width=min(self.image_width, self.raw_frame.shape[1]))
    
    # Show the frame to the screen
    def showFrame(self, frame):
        cv2.imshow(self.image_name, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)



