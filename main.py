import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
# import ImageSegmentation
# import grabcut 
import random

# ROI
REMOVE_EYES = False
FOREHEAD_ONLY = False
ADD_BOX_ERROR = False
USE_SEGMENTATION = False

NUM_ITERATIONS = 0
faceBox = 0 #temp

MIN_FACE_SIZE = 100

WIDTH_FRAC = 0
HEIGHT_FRAC = 0

EYE_LOWER_FRAC = 0.25
EYE_UPPER_FRAC = 0.5

BOX_ERROR_MAX = 0.5

# plotting
FPS = 60
WINDOW_TIME_SEC = 30
WINDOW_NUM_SAMP = int(np.ceil(WINDOW_TIME_SEC * FPS))
MIN_HR = 40.0
MAX_HR = 200.0

# prepare camera capture
RESULTS_SAVE_DIR = "./results/" + ("segmentation/" if USE_SEGMENTATION else "no_segmentation/")
DEFAULT_CAP = "android-1 (1).mp4"
VIDEO_DIR = "./video/"

# GLOBAL VAR
avgRGB_LIST = [] # stores the avg RBG values in each ROI (where analysis is performed)
heartRates = [] # stores hr calculated per seconds 
prevFaceBox = None # stores face box coordinates from previous frame

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

def getBestROI(frame, faceCascade, prevFaceBox):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # choose shade of grey
    faces = faceCascade.detectMultiScale(grey, scaleFactor=1.1, 
        minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE), flags=cv2.CASCADE_SCALE_IMAGE) # detect faces
    
    faceBox = None

    # If no face detected, use ROI from previous frame
    if len(faces) == 0:
        faceBox = prevFaceBox

    # If many faces detected, use one closest to that from previous frame
    elif len(faces) > 1:
        if prevFaceBox is not None:
            # Find closest to previous frame
            minDist = distanceROI(prevFaceBox, face) # compare rest against first distance
            for face in faces:
                if distanceROI(prevFaceBox, face) < minDist:
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
            for i in range(4): # for each four points x1, x2, y1, y2
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

def plotSignals(signals, label):
    seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS) # generate x-axis
    colors = ["r", "g", "b"]
    fig = plt.figure() # make blank figure
    fig.patch.set_facecolor('white')
    for i in range(3):
        plt.plot(seconds, signals[:,i], colors[i]) # plot each signal against seconds
    # label axis
    plt.xlabel('Time (sec)', fontsize=17)
    plt.ylabel(label, fontsize=17)
    # set tick label sizes
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    # display figure
    plt.show()

def plotSpectrum(freqs, powerSpec):
    idx = np.argsort(freqs) # sort freqs
    fig = plt.figure() # make blank figure
    fig.patch.set_facecolor('white')
    for i in range(3):
        plt.plot(freqs[idx], powerSpec[idx,i]) # plot powerSpec against freqs
    # label axis
    plt.xlabel("Frequency (Hz)", fontsize=17)
    plt.ylabel("Power", fontsize=17)
    # set tick label sizes
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    # set x-axis bounds
    plt.xlim([0.75, 4])
    # display figure
    plt.show()

def getHeartRate(windowFrames, lastHR):
    # Normalize dataset to achieve zero mean and unit variance
    mean = np.mean(windowFrames, axis=0)
    std = np.std(windowFrames, axis=0)
    normalizedData = (windowFrames - mean)/std

    # Separating data into 3 independent components using ICA
    ica = FastICA()
    icaSig = ica.fit_transform(normalizedData) #scale test data and learn scaling parameters

    # Find dominant frequency from power spectrum
    powerSpec = np.abs(np.fft.fft(icaSig, axis=0))**2 # use FFT to convert independent components (time-domain signals) into freq domain
    # powerSpec (n,m), n = axis 0 = source signals, m = axis 1 =frequencies
    freq = np.fft.fftfreq(WINDOW_NUM_SAMP, 1.0/FPS)

    # Calculate max heart rate
    maxPowerSpec = np.max(powerSpec, axis=1) # find max powerSpec values in freq axis (col)
    validIdx = np.where(freq >= MIN_HR/60) & (freq <= MAX_HR/60)  # finds indices of freq in range
    
    validPowerSpec = maxPowerSpec(validIdx) # new array with max power for valid freq
    validFreq = freq(validIdx) 
    maxValidPower = np.argmax(validPowerSpec) # find index of max element in validPowerSpec
    heartRate = validFreq(maxValidPower)

    return heartRate

try:
    capFile = sys.argv[1]
except:
    capFile = DEFAULT_CAP
cap = cv2.VideoCapture(VIDEO_DIR + capFile)
# cap = cv2.VideoCapture(1) # captures video from front camera
HAAR_CASCADE = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(HAAR_CASCADE) # face detection trained to detect frontal faces

# capture every frame
while True:
    #while cap.isOpened():
    ret, frame = cap.read() # ret: bool, is camera available; frame: gets next frame
    if not ret:
        break

     # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prevFaceBox, roi = getBestROI(frame, faceCascade, prevFaceBox)
    
    # Add each frames avgColour to list
    if roi is not None and np.size(roi) > 0: # face is detected
        avgColour = np.mean(roi.reshape(-1, roi.shape[-1]), axis=0) # (x,y,z) -> (xy, z), mean calculated for each row (R,G,B)
        avgRGB_LIST.append(avgColour) 

    # Calculate heart rate
    if (len(avgRGB_LIST) >= WINDOW_NUM_SAMP) and (len(avgRGB_LIST) % np.ceil(FPS)==0):
        # windowStartIdx = len(avgRGB_LIST) - WINDOW_NUM_SAMP # index in list avgRGB_LIST where the calculation starts
        # windowFrames = avgRGB_LIST[windowStartIdx : windowStartIdx+WINDOW_NUM_SAMP]
        windowFrames = avgRGB_LIST[-WINDOW_NUM_SAMP:] # retrieves last WINDOW_NUM_SAMP elements
        lastHR = heartRates[-1] if heartRates else None
        heartRates.append(getHeartRate(windowFrames, lastHR))

    if np.ma.is_masked(roi):
        roi = np.where(roi.mask == True, 0, roi)
        cv2.imshow('ROI', roi)
        cv2.waitKey(5)

print(heartRates)
# print (videoFile)
filename = RESULTS_SAVE_DIR + capFile[0:-4]
if ADD_BOX_ERROR:
    filename += "_" + str(BOX_ERROR_MAX)
np.save(filename, heartRates)
cap.release()
cv2.destroyAllWindows()