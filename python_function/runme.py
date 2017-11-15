
from optical_flow_functions import *
from basic_packages import *

# setup video capture
cap = cv2.VideoCapture("../Datasets/MarquesBrownlee.mp4")
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