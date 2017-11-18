'''
  File name: applyGeometricTransformation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for bounding box
    - Input startXs: the x coordinates for all features wrt the first frame
    - Input startYs: the y coordinates for all features wrt the first frame
    - Input newXs: the x coordinates for all features wrt the second frame
    - Input newYs: the y coordinates for all features wrt the second frame
    - Input bbox: corner coordiantes of all detected bounding boxes
    
    - Output Xs: the x coordinates(after eliminating outliers) for all features wrt the second frame
    - Output Ys: the y coordinates(after eliminating outliers) for all features wrt the second frame
    - Output newbbox: corner coordiantes of all detected bounding boxes after transformation
'''

from basic_packages import *
from helper import *

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    #TODO: Your code here

    # number of frames
    num_frames = startXs.shape[0]

    # Init return Xs and Ys
    Xs = []
    Ys = []
    newbbox = []

    for i in xrange(num_frames):
        box = bbox[i] # (4,2)
        newBox = np.zeros(box.shape)
        f = transform.SimilarityTransform()
        src = np.asarray([startXs[i], startYs[i]]).T
        dst = np.asarray([newXs[i], newYs[i]]).T
        if(not f.estimate(src,dst)):
            raise ValueError("The matrix estimation fail") 
        else:  
            H = f.params   
        a_0 = H[0][0]
        b_0 = H[1][0]
        a_1 = H[0][2]
        b_1 = H[1][2]

        newBox[:,0] = a_0 * box[:,0] - b_0 * box[:,1] + a_1
        newBox[:,1] = b_0 * box[:,0] + a_0 * box[:,1] + b_1

        # Deformed box
        newBox = np.round(newBox).astype(int)

        # Boundary
        right = np.amax(newBox[:,1])        
        left = np.amin(newBox[:,1])
        bottom = np.amax(newBox[:,0])
        top = np.amin(newBox[:,0])

        # Regular box
        newBox = np.asarray([[top,left],[top,right],[bottom,left],[bottom,right]])
        newbbox.append(newBox)

        # eliminate out-of-box features
        xys = np.asarray([newXs[i], newYs[i]]).T
        xys[xys[:,0] >= bottom] = -1
        xys[xys[:,0] <= top] = -1
        xys[xys[:,1] >= right] = -1
        xys[xys[:,1] <= left] = -1        
        xys = xys[np.all(xys != -1, axis = 1),:]

        
        # Append into result
        Xs.append(xys[:,0])
        Ys.append(xys[:,1]) 

    # pdb.set_trace()

    Xs = np.asarray(Xs)
    Ys = np.asarray(Ys)
    newbbox = np.asarray(newbbox)
    return Xs, Ys, newbbox


