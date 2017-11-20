'''
  File name: helper.py
  Author:
  Date created:
'''

import numpy as np
from scipy import signal

'''
  File clarification:
  Include any helper function you want for this project such as the 
  video frame extraction, video generation, drawing bounding box and so on.
'''

'''
  Generate one dimension Gaussian distribution
  - input mu: the mean of pdf
  - input sigma: the standard derivation of pdf
  - input length: the size of pdf
  - output: a row vector represents one dimension Gaussian distribution
'''
def GaussianPDF_1D(mu, sigma, length):
  # create an array
  half_len = length / 2

  if np.remainder(length, 2) == 0:
    ax = np.arange(-half_len, half_len, 1)
  else:
    ax = np.arange(-half_len, half_len + 1, 1)

  ax = ax.reshape([-1, ax.size])
  denominator = sigma * np.sqrt(2 * np.pi)
  nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )

  return nominator / denominator

'''
  Generate two dimensional Gaussian distribution
  - input mu: the mean of pdf
  - input sigma: the standard derivation of pdf
  - input row: length in row axis
  - input column: length in column axis
  - output: a 2D matrix represents two dimensional Gaussian distribution
'''
def GaussianPDF_2D(mu, sigma, row, col):
  # create row vector as 1D Gaussian pdf
  g_row = GaussianPDF_1D(mu, sigma, row)
  # create column vector as 1D Gaussian pdf
  g_col = GaussianPDF_1D(mu, sigma, col).transpose()

  return signal.convolve2d(g_row, g_col, 'full')
