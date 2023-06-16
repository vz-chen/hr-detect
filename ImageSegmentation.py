import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn #may need smth more specific
import ImageSegmentation
from Grabcut import grabCut 


# ROI
ITERATIONS = 0

# GLOBAL FUNCTION
ROIavgRBG = [] # stores the avg RBG values in each ROI (where analysis is performed)
heartRates = [] # stores hr calculates 
prevFaceBox = None # stores face box coordinates from previous frame

# prepare camera capture
cap = cv2.VideoCapture(1) # captures video from front camera
HAAR_CASCADE = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(HAAR_CASCADE) # face detection trained to detect frontal faces

# capture every frame
while cap.isOpened():
    ret, frame = cap.read() # ret: bool, is camera available; frame: gets next frame
    if not ret:
        break

# OTHER FUNCTIONS
def imageSegment(image):
    # initial array to store all pixels of image to be labelled fg or bg later
    mask = np.zeros(image.shape[:2], np.uint8) 
    # array of zeros with size as image (height, width, ignoring the third tuple: channel)
    # each element is a 8-bit unsigned binary integer

    bgModel = np.zeros((1,65),np.float64) # use to estimate fg and bg models based on iterative gueseses
    fgModel = np.zeros((1,65),np.float64) 

    cv2.grabCut(image,mask,faceBox,bgModel,fgModel, ITERATIONS, cv2.GC_INIT_WITH_RECT)

    return 0