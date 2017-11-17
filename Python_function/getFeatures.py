'''
  File name: getFeatures.py
  Author:
  Date created:
  '''

'''
  File clarification:
    Detect features within each detected bounding box
    - Input img: the first frame (in the grayscale) of video
    - Input bbox: the four corners of bounding boxes
    - Output x: the x coordinates of features
    - Output y: the y coordinates of features
    '''

from basic_packages import *


def getFeatures(img, bbox):

    x = []
    y = []

    for box in bbox:
        tempx =int((box[2][0]-box[0][0])*0.1)
        tempy =int((box[1][1]-box[0][1])*0.1)
        box_img = img[box[0][0]+tempx:box[2][0]-tempx, 
                            box[0][1]+tempy:box[1][1]-tempy]
        xys = corner_peaks(corner_shi_tomasi(box_img, sigma=1))
        x.append(box[0,0]+xys[0:len(xys),0]+tempx)
        y.append(box[0,1]+xys[0:len(xys),1]+tempy)

    x = np.asarray(x).astype(int)
    y = np.asarray(y).astype(int)
 
    return x, y