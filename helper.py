'''
  File name: helper.py
  Author:
  Date created:
'''

'''
  File clarification:
  Include any helper function you want for this project such as the 
  video frame extraction, video generation, drawing bounding box and so on.
'''
import cv2
import numpy as np
from scipy import signal

def drawBox(img, bbox):
  imgwbox = img.copy()
  for box in bbox:
    l_t_pt = (box[0])[::-1]
    r_b_pt = box[3][::-1]
    cv2.rectangle(imgwbox,tuple(l_t_pt),tuple(r_b_pt),(0,255,0),2)
  return imgwbox

def gaussianPDF_1D(mu, sigma, length):
  half_len = length/2
  if np.remainder(length, 2)==0:
    ax = np.arange(-half_len, half_len, 1)
  else:
    ax = np.arange(-half_len, half_len + 1, 1)
  ax = ax.reshape([-1, ax.size])
  den = sigma * np.sqrt(2 * np.pi)
  nom = np.exp(-np.square(ax - mu) / (2 * sigma * sigma))
  return nom/den

def gaussianPDF(mu, sigma, row, col):
  g_row = gaussianPDF_1D(mu, sigma, row)
  g_col = gaussianPDF_1D(mu, sigma, col).transpose()
  return signal.convolve2d(g_row, g_col, 'full')