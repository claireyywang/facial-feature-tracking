'''
  File name: applyGeometricTransformation.py
  Author:
  Date created:
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf

from detectFace import detectFace
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation

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

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):

    Xs =[]
    Ys =[]
    newbbox =[]
    for i in xrange(len(bbox)):
        box =np.array(bbox[i])
        dist =((startXs[i] - newXs[i]) ** 2 + (startYs[i] - newYs[i]) ** 2) ** (0.5)
        ind =dist < 5
        Xs.append(newXs[i][ind])
        Ys.append(newYs[i][ind])

        trans_f =tf.SimilarityTransform()
        src =np.asarray([startXs[i][ind], startYs[i][ind]]).T
        dst =np.asarray([newXs[i][ind], newYs[i][ind]]).T

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

if __name__ =='__main__':
  cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Medium\StrangerThings.mp4')
  # cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Easy\TheMartian.mp4')
  [retval0, img0] = cap.read(0)
  [retval1, img1] = cap.read(1)
  cap.release()
  # bbox = detectFace(img0)
  # np.save('Bbox', bbox)
  bbox = np.load('.\Bbox.npy')
  [startXs, startYs] = getFeatures(img0, bbox)
  [newXs, newYs] = estimateAllTranslation(startXs, startYs, img0, img1)
  applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)