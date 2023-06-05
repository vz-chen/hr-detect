import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture 
import maxflow

# calculate the number of pixels, flatten 3d array of pixels into a 2d array
def getPixels(pixels3D):
    numPixels = pixels3D.shape[0] * pixels3D.shape[1] # total number of pixels
    pixels = np.empty((numPixels, 3)) # initialize empty array with row size numPixels (total pixels) and 3 cols
    for i in range(3):
        pixels[:,i] = pixels3D[:,:,i].flatten() # : means select all, pixels[:,i] -> select all rows from a col, pixels3D[:,:,i] -> select all row/cols from channel
    return pixels