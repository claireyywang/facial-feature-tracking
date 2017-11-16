'''
  File name: estimateFeatureTranslation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for single features 
    - Input startX: the x coordinate for single feature wrt the first frame
    - Input startY: the y coordinate for single feature wrt the first frame
    - Input Ix: the gradient along the x direction
    - Input Iy: the gradient along the y direction
    - Input img1: the first image frame
    - Input img2: the second image frame
    - Output newX: the x coordinate for the feature wrt the second frame
    - Output newY: the y coordinate for the feature wrt the second frame
'''

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
  #TODO: Your code here
  return newX, newY
  img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  d = img1 - img2;


if __name__ == '__main__':
  # setup video capture
  cap = cv2.VideoCapture("/Users/claraw/Desktop/Feature_Tracking_Optical_Flow/Datasets/Easy/TheMartian.mp4")
  ret,img1 = cap.read()
  ret,img2 = cap.read()
  cap.release()

  bbox = detectFace(img1)
  startX, startY = getFeatures(img1, bbox)
  Ix, Iy = np.gradient(img1)