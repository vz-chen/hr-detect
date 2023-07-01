import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn #may need smth more specific
import ImageSegmentation
# import grabcut 
import random

# ROI
REMOVE_EYES = False
FOREHEAD_ONLY = False
ADD_BOX_ERROR = False

NUM_ITERATIONS = 0
faceBox = 0 #temp

MIN_FACE_SIZE = 100

WIDTH_FRAC = 0
HEIGHT_FRAC = 0

EYE_LOWER_FRAC = 0.25
EYE_UPPER_FRAC = 0.5

BOX_ERROR_MAX = 0.5

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

    # adjust faceBox dimensions (e.g. eye=smaller ROI)
    (x, y, w, h) = faceBox
    widthOffset = int((1 - widthFrac) * w / 2)
    heightOffset = int((1 - heightFrac) * h / 2)
    new_faceBox = (x + widthOffset, y + heightOffset, int(w * widthFrac), int(h * heightFrac))
    (x, y, w, h) = new_faceBox

    # adjust bgMask array, set bg to False and fg to True
    bgMask = np.full(image.shape, True, dtype=bool) #initially change all bgMask elements to true
    bgMask[y:y+h, x:x+w, :] = False # setting pixels of bg as False, so rest is true = fg

    (x, y, w, h) = faceBox

    if REMOVE_EYES:
        bgMask[y + h * EYE_LOWER_FRAC : y + h * EYE_UPPER_FRAC, :] = True
    if FOREHEAD_ONLY:
        bgMask[y + h * EYE_LOWER_FRAC :, :] = True

    roi = np.ma.array(image, mask=bgMask) # Masked array
    return roi

def distanceROI(roi1, roi2):
    squareDistance = 0 # no need to square root for real distance because only used for comparison
    for i in range(len(roi1)): # len is 2
        squareDistance += (roi1[i] - roi2[i])**2 # distance between x and y coords of two points
    return squareDistance

def getBestROI(frame, faceCascade, previousFaceBox):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # choose shade of grey
    faces = faceCascade.detectMultiScale(grey, scaleFactor=1.1, 
        minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE), flags=cv2.cv.CV_HAAR_SCALE_IMAGE) # detect faces
    roi = None
    faceBox = None

    # If no face detected, use ROI from previous frame
    if len(faces) == 0:
        faceBox = previousFaceBox

    # If many faces detected, use one closest to that from previous frame
    elif len(faces) > 1:
        if previousFaceBox is not None:
            # Find closest to previous
            minDist = distanceROI(previousFaceBox, face) # compare rest against first distance
            for face in faces:
                if distanceROI(previousFaceBox, face) < minDist:
                    faceBox = face
        else:
            # Chooses largest box by area (most likely to be true face)
            maxArea = 0
            for face in faces:
                if (face[2] * face[3]) > maxArea:
                    faceBox = face

    # If one face dectected, use face
    else:
        faceBox = faces[0]

    if faceBox is not None:
        if ADD_BOX_ERROR:
            # add margin of error around faceBox coordinates
            noise = []
            for i in range(4):
                noise.append(random.uniform(-BOX_ERROR_MAX, BOX_ERROR_MAX))
            (x, y, w, h) = faceBox
            x1 = x + int(noise[0] * w)
            y1 = y + int(noise[1] * h)
            x2 = x + w + int(noise[2] * w)
            y2 = y + h + int(noise[3] * h)
            faceBox = (x1, y1, x2-x1, y2-y1)

        # Show rectangle
        #(x, y, w, h) = faceBox
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        roi = getROI(frame, faceBox)

    return faceBox, roi
