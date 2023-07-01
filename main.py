import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn #may need smth more specific
import ImageSegmentation
# import grabcut 


# ROI
NUM_ITERATIONS = 0
faceBox = 0 #temp
WIDTH_FRAC = 0
HEIGHT_FRAC = 0

# GLOBAL FUNCTION
ROIavgRBG = [] # stores the avg RBG values in each ROI (where analysis is performed)
heartRates = [] # stores hr calculates 
prevFaceBox = None # stores face box coordinates from previous frame

# prepare camera capture
cap = cv2.VideoCapture(1) # captures video from front camera
HAAR_CASCADE = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(HAAR_CASCADE) # face detection trained to detect frontal faces

# capture every frame
for i in range(20, -1, -1):
    #while cap.isOpened():
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
    fgModel = np.zeros((1,65),np.float64) # np.zeros: return array filled with 0s of size x,y
    # iterate grabCut NUM_ITERATIONS times
    cv2.grabCut(image,mask,faceBox,bgModel,fgModel, NUM_ITERATIONS, cv2.GC_INIT_WITH_RECT)

    bgMask = ((mask == cv2.GC_BGD | mask == cv2.GC_PR_BGD, True, False)).astype('uint8') 
    # define binary array bgMask as the pixels that are 100% bg and prob bg with T or F elements
    bgMask = np.broadcast_to(bgMask[:,:,np.newaxis], image.shape) # match shape of bgMask to image's
    
    return bgMask

def getROI(image, faceBox):

    # get width and height fractions of facebox to specify ROI (frac of face to be in ROI)
    widthFrac = WIDTH_FRAC
    heightFrac = HEIGHT_FRAC

    #adjust faceBox dimensions (e.g. eye=smaller ROI)
    (x,y,w,h) = faceBox
    widthOffset = int((1-widthFrac) * w/2)
    heightOffset = int((1-heightFrac) * h/2)
    new_faceBox = (x+widthOffset,y+heightOffset, int(w*widthFrac), int(h*heightFrac))
    (x,y,w,h) = new_faceBox

    #adjust bgMask array, set bg to False anhd fg to True
    bgMask = np.full(image.shape, True, bool) #initially change all bgMask elements to true
    bgMask[y:y+h , x:x+w , :] = False # setting pixels of bg as False, so rest is true = fg


    return 0

