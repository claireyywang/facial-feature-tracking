'''
  File name: faceTracking.py
  Author:
  Date created:
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectFace import detectFace
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation

'''
  File clarification:
    Generate a video with tracking features and bounding box for face regions
    - Input rawVideo: the video contains one or more faces
    - Output trackedVideo: the generated video with tracked features and bounding box for face regions
'''

def faceTracking(rawVideo):
  cap = cv2.VideoCapture(rawVideo)
  # cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Medium\StrangerThings.mp4')
  # cap = cv2.VideoCapture('.\CIS581Project4PartADatasets\Easy\TheMartian.mp4')

  retval, img0 =cap.read()
  # bbox = detectFace(img0)
  # np.save('Bbox', bbox)
  bbox = np.load('.\Bbox.npy')
  [startXs, startYs] = getFeatures(img0, bbox)
  imgstack =[]
  tempimg0 =img0.copy()
  count =0

  for i in xrange(len(bbox)):
    box = bbox[i]
    cv2.rectangle(tempimg0, (box[0][1], box[0][0]), (box[-1][1], box[-1][0]), (0, 255, 0), 3)
  plt.figure()
  plt.imshow(tempimg0)
  plt.axis('off')
  imgstack.append(tempimg0)
  # cv2.imwrite('img0out.jpg', img0)

  while(retval):
    [retval, img1] = cap.read()
    tempimg1 =img1.copy()

    [newXs, newYs] = estimateAllTranslation(startXs, startYs, img0, img1)
    [Xs, Ys, newbbox] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)
    templen =np.array([])

    for i in xrange(len(Xs)):
      templen =np.append(templen, len(Xs[i]))

    if sum(templen < 10) > 0:
      newbbox = detectFace(img1)
      [Xs, Ys] = getFeatures(img1, newbbox)

    #update for next iteration
    img0 =img1
    startXs =Xs
    startYs =Ys
    bbox =newbbox

    for i in xrange(len(bbox)):
      newbox =newbbox[i]
      cv2.rectangle(tempimg1, (newbox[0][1], newbox[0][0]), (newbox[-1][1], newbox[-1][0]), (0, 255, 0), 3)

    imgstack.append(tempimg1)

    plt.figure()
    # plt.subplot(121)
    # plt.imshow(img1)
    # plt.axis('off')
    # plt.subplot(122)
    plt.imshow(tempimg1)
    plt.axis('off')
    # plt.show()
    count =count +1
    print count

  cap.release()
  cv2.destroyAllWindows()

  # return trackedVideo

if __name__ =='__main__':
  filename ='.\CIS581Project4PartADatasets\Medium\StrangerThings.mp4'
  faceTracking(filename)

