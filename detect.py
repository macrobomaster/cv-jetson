
# Python program for Detection of a  
# specific color(blue here) using OpenCV with Python 
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import numpy as np  
   
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
    help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
    help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# Webcamera no 0 is used to capture the frames 
print("[INFO] sampling THREADED frames from webcam...")
cap = WebcamVideoStream(src=0).start()
fps = FPS().start()


# This drives the program into an infinite loop. 
while(1):        
    # Captures the live stream frame-by-frame 
    frame = cap.read() 
    frame = imutils.resize(frame, width=900)
    # Converts images from BGR to HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    lower_red = np.array([110,50,50]) 
    upper_red = np.array([130,255,255]) 
  
    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask = cv2.inRange(hsv, lower_red, upper_red) 
      
    # The bitwise and of the frame and mask is done so  
    # that only the blue coloured objects are highlighted  
    # and stored in res 
    res = cv2.bitwise_and(frame,frame, mask= mask) 
    cv2.imshow('frame',frame) 
    cv2.imshow('mask',mask) 
    cv2.imshow('res',res) 
    cv2.waitKey(1)
    
    # update the FPS counter
    fps.update()
      
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows() 
  
# release the captured frame 
cap.stop() 