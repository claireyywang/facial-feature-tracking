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
from skimage import transform

from detectFace import detectFace
from getFeatures import getFeatures
from estimateFeatureTranslation import estimateFeatureTranslation
from estimateAllTranslation import estimateAllTranslation

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
  #TODO: Your code here
  Xs = []
  Ys = []
  
  newbbox = []
  for i in xrange(len(bbox)):
    box = np.array(bbox[i])
    startXs[i] = np.array(startXs[i])
    startYs[i] = np.array(startYs[i])
    newXs[i] = np.array(newXs[i])
    newYs[i] = np.array(newYs[i])
    dist = ((startXs[i] - newXs[i]) ** 2 + (startYs[i] - newYs[i]) ** 2) ** (0.5)

    # multiple feature boxes
    idx = np.array(np.where(dist < 5))[0]
    tmpXs = []
    tmpYs = []
    tmpStartXs = []
    tmpStartYs = []
    tmpNewXs = []
    tmpNewYs =[]
    for ind in idx:
      tmpXs.append(newXs[i][ind])
      tmpYs.append(newYs[i][ind])
      tmpStartXs.append(startXs[i][ind])
      tmpStartYs.append(startYs[i][ind])
      tmpNewXs.append(newXs[i][ind])
      tmpNewYs.append(newYs[i][ind])
    Xs.append(tmpXs)
    Ys.append(tmpYs)

    trans_f = transform.SimilarityTransform()
    src = np.asarray([tmpStartXs, tmpStartYs]).T
    dst = np.asarray([tmpNewXs, tmpNewYs]).T

    if not trans_f.estimate(src, dst):
      raise ValueError('transformation function failed')
    else:
      H = trans_f.params

    a0 = H[0][0]
    b0 = H[0][1]
    a1 = H[0][2]
    b1 = H[1][2]

    newbox = np.zeros((box.shape[0], box.shape[1]))
    newbox[:, 0] = a0 * box[:, 0] + b0 * box[:, 1] + a1
    newbox[:, 1] = b0 * box[:, 0] + a0 * box[:, 1] + b1

    box_x1 = int(round(min(newbox[0:2, 0])))
    box_x2 = int(round(max(newbox[2:4, 0])))
    box_y1 = int(round(min(newbox[0, 1], newbox[2, 1])))
    box_y2 = int(round(max(newbox[1, 1], newbox[3, 1])))

    newbox = np.asarray([[box_x1, box_y1], [box_x1, box_y2], [box_x2, box_y1], [box_x2, box_y2]])
    newbbox.append(newbox)

  #     plt.figure(1)
  #     plt.subplot(121)
  #     plt.imshow(img0)
  #     plt.plot(box.T[1], box.T[0], 'ro')
  #     plt.axis('off')
  #     plt.subplot(122)
  #     plt.imshow(img1)
  #     plt.plot(newbox.T[1], newbox.T[0], 'ro')
  #     plt.axis('off')
  # plt.show()

  return Xs, Ys, newbbox


if __name__ == '__main__':
  cap = cv2.VideoCapture("./Datasets/Difficult/StrangerThings.mp4")
  ret,img1 = cap.read()
  ret,img2 = cap.read()
  cap.release()
  tmpimg1 = img1.copy()
  tmpimg2 = img2.copy()

  bbox = detectFace(img1)
  startXs, startYs = getFeatures(img1, bbox)
  newXs, newYs = estimateAllTranslation(startXs, startYs, img1, img2)
  Xs, Ys, newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)

  for box in bbox:
    cv2.rectangle(tmpimg1, (int(box[0][1]), int(box[0][0])), (int(box[-1][1]), int(box[-1][0])), (0, 255, 0), 3)
  plt.figure()
  plt.imshow(tmpimg1)
  for j in range(len(startYs)): 
    plt.plot(startYs[j], startXs[j], 'w+')
  plt.axis('off')
  plt.show()

  for newbox in newbbox:
    cv2.rectangle(tmpimg2, (int(newbox[0][1]), int(newbox[0][0])), (int(newbox[-1][1]), int(newbox[-1][0])), (0, 255, 0), 3)
  plt.figure()
  plt.imshow(tmpimg2)
  for i in range(len(Xs)):  
    plt.plot(Ys[i], Xs[i], 'w+')
  plt.axis('off')
  plt.show()