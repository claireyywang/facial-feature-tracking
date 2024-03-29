'''
  File name: getFeatures.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detect features within each detected bounding box
    - Input img: the first frame (in the grayscale) of video
    - Input bbox: the four corners of bounding boxes
    - Output x: the x coordinates of features
    - Output y: the y coordinates of features
'''
import cv2
import pdb
import numpy as np 
import scipy
import matplotlib.pyplot as plt
from skimage.feature import corner_shi_tomasi, corner_peaks

from detectFace import detectFace
from helper import *
def getFeatures(img, bbox):
  #TODO: Your code here
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_gray = np.array(img_gray)
  x = []
  y = []
  for box in bbox:
    #box = [(y,x),(y,x+w),(y+h,x),(y+h,x+w)]
    tempx =int((box[2][0]-box[0][0])*0.1)
    tempy =int((box[1][1]-box[0][1])*0.1)
    box_img = img_gray[box[0][0]+tempx:box[2][0]-tempx, 
                        box[0][1]+tempy:box[1][1]-tempy]
    xys = corner_peaks(corner_shi_tomasi(box_img, sigma=0.5))
    # plt.figure()
    # plt.imshow(box_img, cmap='gray')
    # plt.axis('off')
    # plt.show()
    x.append(box[0,0]+xys[0:len(xys),0]+tempx)
    y.append(box[0,1]+xys[0:len(xys),1]+tempy)
  x = np.array(x)
  y = np.array(y)
  # print x
  # print y
  # imgwbox = drawBox(img, bbox)
  # plt.figure()
  # plt.imshow(imgwbox)
  # for i in range(len(x)):  
  #   plt.plot(y[i], x[i], 'w+')
  # plt.axis('off')
  # plt.show()
  return x, y

if __name__ == '__main__':
  # setup video capture
  cap = cv2.VideoCapture("./Datasets/Difficult/StrangerThings.mp4")  
  ret,img = cap.read()
  #small = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
  cap.release()
  if ret:
    print ("Frame read %s", ret)
    bbox = detectFace(img)
  else:
    print ("Frame read %s", ret)
  x, y = getFeatures(img, bbox)