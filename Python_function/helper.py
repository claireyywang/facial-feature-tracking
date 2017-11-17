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

def debug_draw(img,X,Y):

	plt.figure()
	img = np.uint8(img)
	plt.imshow(img, cmap='gray')
	# plt.title("{}".format(filename))	
	plt.scatter(x=Y, y=X, c='r', s=3)
	plt.axis('off')
	plt.show()