'''
  File name: estimateAllTranslation.py
  Author:
  Date created:
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectFace import detectFace
from getFeatures import getFeatures
from estimateFeatureTranslation import estimateFeatureTranslation


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

  [Ix, Iy] = np.gradient(gray0)

  newXs =[]
  newYs =[]

  for i in xrange(len(startXs)):
    listXs = []
    listYs = []
    for j in xrange(len(startXs[i])):
      startX =startXs[i][j]
      startY =startYs[i][j]
      [newX, newY] = estimateFeatureTranslation(startX, startY, Ix, Iy, img0, img1)
      listXs.append(newX)
      listYs.append(newY)

    newXs.append(np.asarray(listXs))
    newYs.append(np.asarray(listYs))

  #   plt.figure(1)
  #   plt.subplot(121)
  #   plt.imshow(img0)
  #   # plt.plot(startYs[i], startXs[i], 'ro')
  #   plt.axis('off')
  #   plt.subplot(122)
  #   plt.imshow(img1)
  #   # plt.plot(listYs, listXs, 'ro')
  #   plt.axis('off')
  #
  # plt.show()

  return newXs, newYs

if __name__ =='__main__':
  cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Medium\StrangerThings.mp4')
  # cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Easy\TheMartian.mp4')
  [retval0, img0] = cap.read(0)
  [retval1, img1] = cap.read(1)
  cap.release()
  bbox = detectFace(img0)
  np.save('Bbox', bbox)
  # bbox = np.load('.\Bbox.npy')
  [startXs, startYs] = getFeatures(img0, bbox)
  estimateAllTranslation(startXs, startYs, img0, img1)