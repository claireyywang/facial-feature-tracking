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

  fullname =rawVideo.split('\\')
  videoname =str(fullname[-1]).split('.')
  outdir ='.\\video_out\\' + videoname[0] +'.avi'
  retval, img0 =cap.read()
  bbox = detectFace(img0)
  # np.save('Bbox', bbox)
  # bbox = np.load('.\Bbox.npy')
  [startXs, startYs] = getFeatures(img0, bbox)
  # imgstack =[]
  tempimg0 =img0.copy()
  count =0

  height, width, layers = img0.shape
  fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
  video = cv2.VideoWriter(outdir, fourcc, 20, (width, height), True)

  for i in xrange(len(bbox)):
    box = bbox[i]
    cv2.rectangle(tempimg0, (box[0][1], box[0][0]), (box[-1][1], box[-1][0]), (0, 255, 0), 3)
  # plt.figure()
  # plt.imshow(tempimg0)
  # plt.axis('off')
  # imgstack.append(tempimg0)
  video.write(tempimg0)

  while(cap.isOpened()):
    [retval, img1] = cap.read()

    if retval ==True:
      tempimg1 =img1.copy()

      [newXs, newYs] = estimateAllTranslation(startXs, startYs, img0, img1)
      [Xs, Ys, newbbox] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)
      templen =np.array([])

      for i in xrange(len(Xs)):
        templen =np.append(templen, len(Xs[i]))

      if sum(templen < 10) > 0: ########if box size is too big, redetect the face
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

      # imgstack.append(tempimg1)
      video.write(tempimg1)
      # plt.figure()
      # plt.subplot(121)
      # plt.imshow(img1)
      # plt.axis('off')
      # plt.subplot(122)
      # plt.imshow(tempimg1)
      # plt.axis('off')
      # plt.show(block=False)
      # plt.pause(0.0001)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

      count =count +1
      print count

    else:
      break

  cap.release()
  video.release()
  cv2.destroyAllWindows()

  return video

if __name__ =='__main__':
  # filename ='.\CIS581Project4PartADatasets\Medium\StrangerThings.mp4'
  filename ='.\CIS581Project4PartADatasets\Easy\TheMartian.mp4'
  faceTracking(filename)

