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

def detectFace(img):
  #TODO: Your code here 
  # read image to array
  cascPath = 'haarcascade_frontalface_default.xml'
  faceCascade = cv2.CascadeClassifier(cascPath)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Detect faces in the image
  faces = faceCascade.detectMultiScale(
  	gray,
  	#scale factor decides the accuracy of detection
  	scaleFactor=1.3,
  	minNeighbors=5,
  	minSize=(30, 30),
  	flags = cv2.CASCADE_SCALE_IMAGE
  	)
  bbox = []
  # Draw a rectangle around the face
  for (x, y, w, h) in faces:
  	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
  	print ([[x,y], [x+w,y], [x,y+h],[x+w,y+h]])
  	bbox.append([[x,y], [x+w,y], [x,y+h],[x+w,y+h]])
  #bbox = np.array(bbox)
  #cv2.imshow("Faces found" ,img)
  cv2.imwrite("result", img)
  #click on any key to terminate display 
  #cv2.waitKey(0)
  return bbox