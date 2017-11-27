'''
  File name: estimateFeatureTranslation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for single features 
    - Input startX: the x coordinate for single feature wrt the first frame
    - Input startY: the y coordinate for single feature wrt the first frame
    - Input Ix: the gradient along the x direction
    - Input Iy: the gradient along the y direction
    - Input img1: the first image frame
    - Input img2: the second image frame
    - Output newX: the x coordinate for the feature wrt the second frame
    - Output newY: the y coordinate for the feature wrt the second frame
'''
import cv2
import pdb
import numpy as np 
import scipy
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import inv

from detectFace import detectFace
from getFeatures import getFeatures
from helper import *

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
  #TODO: Your code here
  It = img2-img1

  # apply gaussian kernel 
  gaussian = gaussianPDF(0,1,3,3)
  Ixx = signal.convolve2d(Ix, gaussian, 'same')
  Iyy = signal.convolve2d(Iy, gaussian, 'same')

  Ix_window = Ix[startX-5:startX+5, startY-5:startY+5]
  Iy_window = Iy[startX-5:startX+5, startY-5:startY+5]   
  IxIx = np.sum(np.dot(Ix_window, Ix_window))
  IxIy = np.sum(np.dot(Ix_window, Iy_window))
  IyIy = np.sum(np.dot(Iy_window, Iy_window))
  A_square = np.asarray([[IxIx,IxIy],[IxIy,IyIy]])

  It_window = It[startX-5:startX+5, startY-5:startY+5]
  Ixx_window = Ixx[startX-5:startX+5, startY-5:startY+5]
  Iyy_window = Iyy[startX-5:startX+5, startY-5:startY+5]
  IxIt = -np.sum(np.dot(Ixx_window, It_window))
  IyIt = -np.sum(np.dot(Iyy_window, It_window))
  Ab = np.asarray([IxIt, IyIt])
  uv = np.dot(inv(A_square), Ab)
  dx = uv[0]
  dy = uv[1]
  newX = int(round(startX+dx))
  newY = int(round(startY+dy))
  
  return newX, newY

if __name__ == '__main__':
  # setup video capture
  cap = cv2.VideoCapture("./Datasets/Easy/MarquesBrownlee.mp4")
  ret,img1 = cap.read()
  ret,img2 = cap.read()
  cap.release()

  img1 = np.array(img1)
  img2 = np.array(img2)

  bbox = detectFace(img1)
  x,y = getFeatures(img1, bbox)
  img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  Ix, Iy = np.gradient(img1_gray)
  Ix = np.array(Ix)
  Iy = np.array(Iy)

  newXs = []
  newYs = []
  for i in range(len(x)):
    for j in range(len(x[i])):
      startX = x[i][j]
      startY = y[i][j]
      newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1_gray, img2_gray)
      newXs.append(newX)
      newYs.append(newY)

  print len(newXs)
  print len(newYs)

  plt.figure()
  plt.imshow(img2_gray, cmap='gray')
  plt.plot(newYs, newXs, 'w+')
  plt.axis('off')
  plt.show()