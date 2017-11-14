from detectFace import detectFace
import cv2
import numpy as np

# setup video capture

cap = cv2.VideoCapture("/Users/claraw/Desktop/Feature_Tracking_Optical_Flow/Datasets/Easy/MarquesBrownlee.mp4")
ret,img = cap.read()
if ret:
	print ("Frame read %s", ret)
	detectFace(img) 
else:
	print ("Frame read %s", ret)
	
"""
image = cv2.imread('/Datasets/Easy/frame0.jpg')
detectFace(image)
"""