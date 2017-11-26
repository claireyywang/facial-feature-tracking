'''
  File name: faceTracking.py
  Author:
  Date created:
'''

'''
  File clarification:
    Generate a video with tracking features and bounding box for face regions
    - Input rawVideo: the video contains one or more faces
    - Output trackedVideo: the generated video with tracked features and bounding box for face regions
'''
import numpy as np 
import cv2
import matplotlib.pyplot as plt

from detectFace import detectFace
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation

# def faceTracking(rawVideo):
#   #TODO: Your code here
#   ret,img1 = rawVideo.read()
#   bbox = detectFace(img1)
  
#   while ret:
#     startXs, startYs = getFeatures(img1, bbox)
#     newXs, newYs = estimateAllTranslation(startXs, startYs, img1, img2)
#     Xs, Ys, newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)
#     bbox = newbbox
#     startXs = Xs
#     startYs = Ys
#   return trackedVideo

def faceTracking(rawVideo):
  cap = cv2.VideoCapture(rawVideo)
  retval, img0 = cap.read()

  videoname = 'outputvideo'
  outdir ='.\\' + videoname +'.avi'
  height, width, layers = img0.shape
  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  video = cv2.VideoWriter(outdir, fourcc, 20, (width, height), True)

  bbox = detectFace(img0)
  startXs, startYs = getFeatures(img0, bbox)
  tmpimg0 = img0.copy()
  count = 0

  for box in bbox:
    cv2.rectangle(tmpimg0, (int(box[0][1]), int(box[0][0])), (int(box[-1][1]), int(box[-1][0])), (0, 255, 0), 3)
  plt.figure()
  plt.imshow(tmpimg0)
  plt.plot(startYs, startXs, 'w+')
  plt.axis('off')
  plt.show()
  # imgstack.append(tempimg0)
  video.write(tempimg0)

  import pdb; pdb.set_trace()

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
        # import pdb; pdb.set_trace()
        cv2.rectangle(tempimg1, (int(newbox[0][1]), int(newbox[0][0])), (int(newbox[-1][1]), int(newbox[-1][0])), (0, 255, 0), 3)

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

      count = count +1
      print count

    else:
      break

  cap.release()
  video.release()
  cv2.destroyAllWindows()

  return video
if __name__ == '__main__':
  rawvideo = "/Users/claraw/Desktop/Feature_Tracking_Optical_Flow/Datasets/Easy/MarquesBrownlee.mp4"
  faceTracking(rawvideo)