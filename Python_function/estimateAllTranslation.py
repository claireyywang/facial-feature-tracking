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

from basic_packages import *
from helper import *

# only function related to the implement function
from estimateFeatureTranslation import estimateFeatureTranslation


def estimateAllTranslation(startXs, startYs, img1, img2):
    #TODO: Your code here
    # number of frames
    num_frames = startXs.shape[0]


    # gradient of img1
    grad_img1_x, grad_img2_y = np.gradient(img1)

    # initialized return array
    newXs = []
    newYs = []

    # update feature translation
    for i in xrange(num_frames):
        xs = startXs[i]
        ys = startYs[i]
        feature_num = xs.size
        x = []
        y = []
        for j in xrange(feature_num):
            startX, startY = xs[j], ys[j]
            newX, newY = estimateFeatureTranslation(startX, startY, grad_img1_x, grad_img2_y, img1, img2)
            x.append(newX)
            y.append(newY)
        # for debug
        # debug_draw(img2,x,y)     
        
        if(len(x) != 0):
            newXs.append(x)
            newYs.append(y)
    newXs = np.asarray(newXs)
    newYs = np.asarray(newYs)
    return newXs, newYs