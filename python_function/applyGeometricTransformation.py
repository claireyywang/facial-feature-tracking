'''
  File name: applyGeometricTransformation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for bounding box
    - Input startXs: the x coordinates for all features wrt the first frame
    - Input startYs: the y coordinates for all features wrt the first frame
    - Input newXs: the x coordinates for all features wrt the second frame
    - Input newYs: the y coordinates for all features wrt the second frame
    - Input bbox: corner coordiantes of all detected bounding boxes
    
    - Output Xs: the x coordinates(after eliminating outliers) for all features wrt the second frame
    - Output Ys: the y coordinates(after eliminating outliers) for all features wrt the second frame
    - Output newbbox: corner coordiantes of all detected bounding boxes after transformation
'''
import cv2
import pdb
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from skimage.feature import corner_shi_tomasi, corner_peaks
from skimage import transform

from detectFace import detectFace
from getFeatures import getFeatures
from estimateFeatureTranslation import estimateFeatureTranslation
from estimateAllTranslation import estimateAllTranslation

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
  #TODO: Your code here
  newbbox = []
  Xs = []
  Ys = []
  for i in range(0, len(bbox)):
    box = bbox[i]
    prev_xs = startXs[i]
    prev_ys = startYs[i]
    cur_xs = newXs[i]
    cur_ys = newYs[i]
    
    trans_f = transform.SimilarityTransform()
    src = np.asarray([prev_xs, prev_ys]).T
    dst = np.asarray([cur_xs, cur_ys]).T

    if not trans_f.estimate(src, dst):
      raise ValueError('transformation function failed')
    else:
      H = trans_f.params
    
    a0 = H[0][0]
    b0 = H[0][1]
    a1 = H[0][2]
    b1 = H[1][2]

    newbox = np.zeros(box.shape)
    newbox[:,0] = a0*box[:,0] + b0*box[:,1] + a1
    newbox[:,1] = b0*box[:,0] + a0*box[:,1] + b1

    box_x1 = min(newbox[0:2,0])
    box_x2 = max(newbox[2:4,0])
    box_y1 = min(newbox[0,1], newbox[2,1])
    box_y2 = max(newbox[1,1], newbox[3,1])
    
    newbox = np.asarray([[box_x1, box_y1],[box_x1, box_y2],[box_x2, box_y1],[box_x2, box_y2]])
    newbbox.append(newbox)

    diff = (prev_xs - cur_xs)**2 + (prev_ys - cur_ys)**2
    idx = np.where(diff <= 16)
    xys = np.asarray([cur_xs[idx], cur_ys[idx]]).T

    # eliminate out-of-box features
    xys[xys[:,0] < box_x1] = -1
    xys[xys[:,0] > box_x2] = -1
    xys[xys[:,1] < box_y1] = -1
    xys[xys[:,1] > box_y2] = -1        
    xys = xys[np.all(xys != -1, axis = 1),:]

    Xs.append(np.asarray(xys[:,0]))
    Ys.append(np.asarray(xys[:,1]))

  Xs = np.asarray(Xs)
  Ys = np.asarray(Ys)
  # print Xs
  # print Ys
  # print newbbox
  return Xs, Ys, newbbox

if __name__ == '__main__':
  cap = cv2.VideoCapture("/Users/claraw/Desktop/Feature_Tracking_Optical_Flow/Datasets/Easy/TheMartian.mp4")
  ret,img1 = cap.read()
  ret,img2 = cap.read()
  cap.release()

  bbox = detectFace(img1)
  startXs, startYs = getFeatures(img1, bbox)
  newXs, newYs = estimateAllTranslation(startXs, startYs, img1, img2)
  Xs, Ys, newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)