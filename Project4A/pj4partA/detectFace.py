'''
  File name: detectFace.py
  Author:
  Date created:
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
  File clarification:
    Detect or hand-label bounding box for all face regions
    - Input img: the first frame of video
    - Output bbox: the four corners of bounding boxes for all detected faces
'''

def detectFace(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # haar_face =cv2.CascadeClassifier('.\Face_classfier\haarcascade_profileface.xml')
  haar_face = cv2.CascadeClassifier('.\Face_classfier\haarcascade_frontalface_alt.xml')
  nfaces =haar_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

  # plt.figure()
  # plt.imshow(img)
  # plt.axis('off')
  # plt.show() # this is used to check for faces

  bbox = []
  for (x, y, w, h) in nfaces:
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # bbox.append([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    bbox.append([[y, x], [y, x + w], [y + h, x], [y + h, x + w]])

  # cv2.imshow('img', img)
  cv2.destroyAllWindows()

  return bbox

if __name__ == '__main__':
  # cap =cv2.VideoCapture('.\CIS581Project4PartADatasets\Medium\StrangerThings.mp4')
  cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Easy\TheMartian.mp4')
  # cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Medium\TyrionLannister.mp4')
  # cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Easy\TheMartian.mp4')
  # cap =cv2.VideoCapture('C:\Users\Yuemeng\Desktop\school_hw\CIS 581\Project4\CIS581Project4PartA\pj4partA\StrangerThings.mp4')
  retval, img =cap.read()
  # cap.release()
  detectFace(img)