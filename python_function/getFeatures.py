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
import numpy as np
from skimage.feature import corner_shi_tomasi, corner_peaks
from detectFace import detectFace
import matplotlib.pyplot as plt

def getFeatures(img, bbox):
  #TODO: Your code here
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_gray = np.array(img_gray)
  x = np.array([])
  y = np.array([])

  for box in bbox:
    #box = [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]
    box_img = img_gray[box[0,1]:box[2,1]+1, box[0,0]:box[1,0]+1]
    xys = corner_peaks(corner_shi_tomasi(box_img, sigma=1))
    #plt.figure()
    #plt.imshow(box_img, cmap='gray')
    #plt.axis('off')
    #plt.show()
    y = np.concatenate((y, box[0,1]+ xys[0:len(xys), 0]))
    x = np.concatenate((x, box[0,0]+ xys[0:len(xys), 1]))
  plt.figure()
  plt.imshow(img_gray, cmap='gray')
  plt.plot(x, y, 'r.')
  plt.axis('off')
  plt.show()
  return x, y

if __name__ == '__main__':
  # setup video capture
  cap = cv2.VideoCapture("/Users/claraw/Desktop/Feature_Tracking_Optical_Flow/Datasets/Easy/TheMartian.mp4")
  ret,img = cap.read()
  cap.release()
  if ret:
    print ("Frame read %s", ret)
    bbox = detectFace(img)
  else:
    print ("Frame read %s", ret)

  x, y = getFeatures(img, bbox)