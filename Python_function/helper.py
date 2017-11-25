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


from basic_packages import *

'''
debug_draw:
	For debug and ouput point and image
	input filename can be display on title
'''


def debug_draw(img,X=None,Y=None,filename=None):

	plt.figure()
	img = np.uint8(img)
	if(filename is not None):
		plt.title("{}".format(filename))	
	plt.imshow(img, cmap='gray')
	# plt.title("{}".format(filename))	
	if(X is not None and Y is not None):
		plt.scatter(x=Y, y=X, c='r', s=3)
	plt.axis('off')
	plt.show()



'''
draw_box:
	Draw every box on the image
'''
def draw_boxes(img,bbox):
	img_output = copy.copy(img)
	for box in bbox:
		l_t_pt = (box[0])[::-1]
		r_b_pt = box[3][::-1]
		cv2.rectangle(img_output,tuple(l_t_pt),tuple(r_b_pt),(0,255,0),2)
	
	# # Showing the oupout
	# cv2.imshow("Faces found" ,img_output)
	
	return img_output


'''
  Generate two dimensional Gaussian distribution
  - input mu: the mean of pdf
  - input sigma: the standard derivation of pdf
  - input row: length in row axis
  - input column: length in column axis
  - output: a 2D matrix represents two dimensional Gaussian distribution
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

def GaussianPDF_2D(mu, sigma, row, col):
  # create row vector as 1D Gaussian pdf
  g_row = GaussianPDF_1D(mu, sigma, row)
  # create column vector as 1D Gaussian pdf
  g_col = GaussianPDF_1D(mu, sigma, col).transpose()

  return signal.convolve2d(g_row, g_col, 'full')




