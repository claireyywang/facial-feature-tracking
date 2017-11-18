'''
  File name: estimateAllTranslation.py
  Author:
  Date created:
'''
'''
  File clarification:
    Estimate the translation for all features for each bounding box as well as its four corners
    - Input startXs: all x coordinates for features wrt the first frame
    - Input startYs: all y coordinates for features wrt the first frame
    - Input img1: the first image frame
    - Input img2: the second image frame
    - Output newXs: all x coordinates for features wrt the second frame
    - Output newYs: all y coordinates for features wrt the second frame
'''
import cv2
import pdb
import numpy as np 
import scipy
import matplotlib
import matplotlib.pyplot as plt
from skimage.feature import corner_shi_tomasi, corner_peaks

from detectFace import detectFace
from getFeatures import getFeatures
from estimateFeatureTranslation import estimateFeatureTranslation
def estimateAllTranslation(startXs, startYs, img1, img2):
  img1 = np.array(img1)
  img2 = np.array(img2)
  img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  Ix, Iy = np.gradient(img1_gray)
  Ix = np.array(Ix)
  Iy = np.array(Iy)
  newXs = []
  newYs = []
  for i in range(0, len(startXs)):
      tmpXs = []
      tmpYs = []
      for j in range(0, len(startXs[i])):
          startX = startXs[i][j]
          startY = startYs[i][j]
          newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1_gray, img2_gray)
          tmpXs.append(newX)
          tmpYs.append(newY)
      newXs.append(tmpXs)
      newYs.append(tmpYs)

  newXs = np.asarray(newXs)
  newYs = np.asarray(newYs)
  return newXs, newYs

if __name__ == '__main__':
  # setup video capture
  cap = cv2.VideoCapture("/Users/claraw/Desktop/Feature_Tracking_Optical_Flow/Datasets/Easy/TheMartian.mp4")
  ret,img1 = cap.read()
  ret,img2 = cap.read()
  cap.release()

  bbox = detectFace(img1)
  startXs, startYs = getFeatures(img1, bbox)
  newXs, newYs = estimateAllTranslation(startXs, startYs, img1, img2)

  