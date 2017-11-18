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


def debug_draw(img,X,Y,filename=None):

	plt.figure()
	img = np.uint8(img)
	if(filename is not None):
		plt.title("{}".format(filename))	
	plt.imshow(img, cmap='gray')
	# plt.title("{}".format(filename))	
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







