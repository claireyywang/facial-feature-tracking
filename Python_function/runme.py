
from optical_flow_functions import *
from basic_packages import *



def main():
	# setup video capture
	vidcap = cv2.VideoCapture('../Datasets/Easy/TheMartian.mp4')
	success = True
	pre_img = None
	while success:
		success,cur_img = vidcap.read()
	    #convert to gray scale
		cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
		cur_gray = np.asarray(cur_gray)
  		# first image 
  		if(pre_img is None):
  			bbox = detectFace(cur_gray)
			x, y = getFeatures(cur_gray,bbox)
			pre_img = cur_img
			# plt.figure()
			# plt.imshow(cur_img, cmap='gray')
			# plt.axis('off')
			# plt.show()
			# pre_img = cur_img

		# second image
		else:
			pre_gray = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
			pre_gray = np.asarray(pre_gray)
			
			pre_img_gx, pre_img_gy = np.gradient(pre_gray)




			# plt.figure()
			# plt.subplot(2,2,1),plt.imshow(pre_img_gx, cmap='gray')
			# plt.subplot(2,2,2),plt.imshow(pre_img_gy, cmap='gray')
			# plt.axis('off')
			# plt.show()			

			break

  		if cv2.waitKey(10) == 27:  # exit if Escape is hit
			break
	
if __name__ == "__main__":
	main()