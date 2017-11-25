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

from basic_packages import *
from helper import *

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    #                   IxIx IxIy   u          Itx   
    # Apply the formula                 =  -
    #                   IyIx IyIy   v          Ity
    # For each x and y, create a patch of window 10*10)

    m, n = img1.shape

    # calculate temperate gradient
    It = img2 - img1;

    # Convolute with gaussian functions
    gaussian = GaussianPDF_2D(0, 1, 3, 3) #3 is for martins
    Ix_g = signal.convolve2d(Ix, gaussian, 'same')
    Iy_g = signal.convolve2d(Iy, gaussian, 'same')

    # # Padding with mirror length = 5
    # Ix_pad = np.pad(Ix_g,5,'symmetric')
    # Iy_pad = np.pad(Iy_g,5,'symmetric')
    # It_pad = np.pad(It,5,'symmetric')

    patch_Ix = Ix[startX-5:startX+5,startY-5:startY+5]
    patch_Iy = Iy[startX-5:startX+5,startY-5:startY+5]
    patch_It = It[startX-5:startX+5,startY-5:startY+5]
    patch_Ix_g = Ix_g[startX-5:startX+5,startY-5:startY+5]
    patch_Iy_g = Iy_g[startX-5:startX+5,startY-5:startY+5]

    IxIx = np.sum(np.dot(patch_Ix,patch_Ix))
    IxIy = np.sum(np.dot(patch_Ix,patch_Iy))
    IyIy = np.sum(np.dot(patch_Iy,patch_Iy))
    ItIx = np.sum(np.dot(patch_Ix_g,patch_It))
    ItIy = np.sum(np.dot(patch_Iy_g,patch_It))

    # matrix
    H_inv = np.linalg.inv(np.asarray(([IxIx,IxIy],[IxIy,IyIy])))
    b  = np.asarray([ItIx,ItIy])

    # compute u v 
    A   = -np.dot(H_inv,b)
    u,v = A[0],A[1]
    newX = int(round(startX+u))
    newY = int(round(startY+v))
    
    # # Interpolation
    # meshY,meshX = np.meshgrid(np.arange(0,10),np.arange(0,10))
    # meshX = meshX.flatten()
    # meshY = meshY.flatten()
    # update_x = meshX+u
    # update_y = meshY+v
    # f_x = interpolate.interp2d(meshX,meshY,update_x)
    # f_y = interpolate.interp2d(meshX,meshY,update_y)
    # # round down
    # newX = int(round(startX+f_x(0,0)))
    # newY = int(round(startY+f_y(0,0)))
  

    return newX, newY