'''
  File name: detectFace.py
  Author: 
  Date created:
'''

'''
  File clarification:
    Detect or hand-label bounding box for all face regions
    - Input img: the first frame of video
    - Output bbox: the four corners of bounding boxes for all detected faces
'''
import cv2
import pdb
import numpy as np 
import scipy
import matplotlib
import matplotlib.pyplot as plt
from skimage.feature import corner_shi_tomasi, corner_peaks

def detectFace(img):
  #TODO: Your code here 
  # read image to array
  casc = 'haarcascade_frontalface_alt.xml'
  faceCascade = cv2.CascadeClassifier(casc)
  #convert to gray scale 
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Detect faces in the image
  faces = faceCascade.detectMultiScale(
  	gray,
  	#scale factor decides the accuracy of detection
  	scaleFactor=2,
  	minNeighbors=5,
  	minSize=(30, 30),
  	flags = cv2.CASCADE_SCALE_IMAGE
  	)
  bbox = []
  # Draw a rectangle around the face
  for (x, y, w, h) in faces:
  	#cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
  	#print ([(y,x),(y,x+w),(y+h,x),(y+h,x+w)])
  	bbox.append([(y,x),(y,x+w),(y+h,x),(y+h,x+w)])
  bbox = np.array(bbox)
  #cv2.imshow("Faces found" ,img)
  #cv2.imwrite("result.jpg", img)
  #click on any key to terminate display 
  #cv2.waitKey(0)
  return bbox

if __name__ == '__main__':
  # setup video capture
  cap = cv2.VideoCapture("/Users/claraw/Desktop/Feature_Tracking_Optical_Flow/Datasets/Easy/MarquesBrownlee.mp4")
  ret,img = cap.read()
  cap.release()
  if ret:
    print ("Frame read %s", ret)
    bbox = detectFace(img)
  else:
    print ("Frame read %s", ret)