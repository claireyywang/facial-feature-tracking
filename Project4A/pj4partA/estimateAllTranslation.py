'''
  File name: estimateAllTranslation.py
  Author:
  Date created:
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import interpolate
from detectFace import detectFace
from getFeatures import getFeatures


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

def estimateAllTranslation(startXs, startYs, img0, img1):
  gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  plt.figure(0)
  plt.imshow(gray0, cmap='gray')
  plt.axis('off')
  plt.figure(1)
  plt.imshow(gray1, cmap='gray')
  plt.axis('off')
  plt.show()


  check =1
  # return newXs, newYs

if __name__ =='__main__':
  cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Medium\StrangerThings.mp4')
  [retval0, img0] = cap.read(0)
  [retval1, img1] = cap.read(1)
  cap.release()
  # bbox = detectFace(img0)
  # np.save('Bbox', bbox)
  bbox = np.load('.\Bbox.npy')
  [startXs, startYs] = getFeatures(img0, bbox)
  estimateAllTranslation(startXs, startYs, img0, img1)