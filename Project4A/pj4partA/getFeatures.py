'''
  File name: getFeatures.py
  Author:
  Date created:
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectFace import detectFace
from skimage.feature import corner_shi_tomasi, corner_peaks

'''
  File clarification:
    Detect features within each detected bounding box
    - Input img: the first frame (in the grayscale) of video
    - Input bbox: the four corners of bounding boxes
    - Output x: the x coordinates of features [this is corresponding to row]
    - Output y: the y coordinates of features [this is corresponding to column]
'''

def getFeatures(img, bbox):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  x =[]
  y =[]

  for i in xrange(len(bbox)):
    tempx =int((bbox[i][2][0]-bbox[i][0][0])*0.1)
    tempy =int((bbox[i][1][1]-bbox[i][0][1])*0.1)
    tempbox =gray[bbox[i][0][0]+tempx:bbox[i][2][0]-tempx, bbox[i][0][1]+tempy:bbox[i][1][1]-tempy]
    tempcor =corner_shi_tomasi(tempbox, sigma=1)
    temppeak =corner_peaks(tempcor)
    # plt.figure()
    # plt.imshow(tempbox, cmap='gray')
    # plt.axis('off')
    # plt.show() # this is used to check for faces
    x.append(temppeak[:,0]+bbox[i][0][0]+tempx)
    y.append(temppeak[:,1]+bbox[i][0][1]+tempy)
    # plt.figure(1) #this is to visualize the output corner peak
    # plt.imshow(gray, cmap='gray')
    # plt.plot(y[i], x[i], 'ro')
    # plt.axis('off')

  # plt.show()

  return x, y

if __name__=='__main__':
  cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Easy\TheMartian.mp4')
  retval, img = cap.read()
  cap.release()
  bbox = detectFace(img)
  # np.save('Bbox', bbox)
  # bbox = np.load('.\Bbox.npy')
  getFeatures(img, bbox)