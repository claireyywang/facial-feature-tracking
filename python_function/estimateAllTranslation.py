'''
  File name: estimateAllTranslation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for all features for each bounding box as well as its four corners
    - Input startXs: all x coordinates for features wrt the first frame
    - Input startYs: all y coordinates for features wrt the first frame
    - Input img1: the first image frame
    - Input img2: the second image frame
    - Output newXs: all x coordinates for features wrt the second frame
    - Output newYs: all y coordinates for features wrt the second frame
'''

def estimateAllTranslation(startXs, startYs, img1, img2):
  #TODO: Your code here
  

  img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  Ix, Iy = np.gradient(img1_gray)
  Ix = np.array(Ix)
  Iy = np.array(Iy)
  startX = x[0][0]
  startY = y[0][0]
  newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1_gray, img2_gray)
  return newXs, newYs

if __name__ == '__main__':   
  cap = cv2.VideoCapture("/Users/claraw/Desktop/Feature_Tracking_Optical_Flow/Datasets/Easy/TheMartian.mp4")
  ret,img1 = cap.read()
  ret,img2 = cap.read()
  cap.release()

  img1 = np.array(img1)
  img2 = np.array(img2)

  bbox = detectFace(img1)
  x,y = getFeatures(img1, bbox)