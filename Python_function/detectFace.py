'''
  File name: detectFace.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detect or hand-label bounding box for all face regions
    - Input img: the first frame of video (gray)
    - Output bbox: the four corners of bounding boxes for all detected faces
'''

from basic_packages import *


def detectFace(img):
    #TODO: Your code here
    # read image to array
    casc = '../Haarcascade_frontalface/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(casc)


    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        img,
        #scale factor decides the accuracy of detection
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )

    bbox = []
    # Draw a rectangle around the face
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # print ([[x,y], [x+w,y], [x,y+h],[x+w,y+h]])
        bbox.append([(y,x),(y,x+w),(y+h,x),(y+h,x+w)])
    bbox = np.array(bbox)
    #cv2.imshow("Faces found" ,img)
    # cv2.imwrite("../Images/result.jpg", img)
    #click on any key to terminate display
    #cv2.waitKey(0)
    return bbox