'''
  File name: faceTracking.py
  Author:
  Date created:
'''

'''
  File clarification:
    Generate a video with tracking features and bounding box for face regions
    - Input rawVideo: the video contains one or more faces
    - Output tracked_video: the generated video with tracked features and bounding box for face regions
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

from detectFace import detectFace
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation
from helper import *

def faceTracking(rawVideo):
  cap = cv2.VideoCapture(rawVideo)
  output = None
  pre_img = None

  # first frame
  ret, cur_img = cap.read()
  bbox = detectFace(cur_img)
  startXs, startYs = getFeatures(cur_img, bbox)

  # initialize video writer 
  h, w, l = cur_img.shape
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')

  # change tracked_video name for each run 
  tracked_video = './Output_Video/tracked_video2.m4v'
  output = cv2.VideoWriter(tracked_video, fourcc, 20, (w, h), True)

  # draw box on first frame
  imgwbox = drawBox(cur_img, bbox)
  output.write(imgwbox)
  pre_img = cur_img

  count = 0
  while(cap.isOpened()):
    ret, cur_img = cap.read()

    if not ret:
      break

    newXs, newYs = estimateAllTranslation(startXs, startYs, pre_img, cur_img)
    Xs, Ys, newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)

    box_features = np.array([])
    for i in range(len(Xs)):
      box_features = np.append(box_features, len(Xs[i]))

    print sum(box_features)
    if sum(box_features) < 10:
      newbbox = detectFace(cur_img)
      Xs, Ys = getFeatures(cur_img, newbbox)

    pre_img = cur_img
    startXs = Xs
    startYs = Ys
    bbox = newbbox

    imgwbox = drawBox(cur_img, bbox)
    output.write(imgwbox)

    # print video record
    print ('{} frame finished').format(count)
    count += 1
    
  # close video writer
  cv2.destroyAllWindows()
  cap.release()
  output.release()

  return tracked_video
if __name__ == '__main__':
    rawvideo = "./Datasets/Difficult/StrangerThings.mp4"
    faceTracking(rawvideo)
