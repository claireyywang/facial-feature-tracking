
from optical_flow_functions import *
from basic_packages import *
from helper import *


def main():
	# setup video capture
	vidcap = cv2.VideoCapture('../Datasets/Easy/TheMartian.mp4')
	success = True
	# the previous img
	pre_img = None

	# the feature_x and feature_y in previous img  
	startX = None 
	startY = None
	while success:
		success,cur_img = vidcap.read()
	    #convert to gray scale
  		# first image 
  		if(pre_img is None):
  			cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
			cur_gray = np.asarray(cur_gray)
  			bbox = detectFace(cur_gray)
			startXs, startYs = getFeatures(cur_gray,bbox)
			pre_img = cur_img

		# after second image
		else:
			# change the previous img into gray 
			pre_gray = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
			pre_gray = np.asarray(pre_gray)
			

			# change the currnet img into gray 
  			cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
			cur_gray = np.asarray(cur_gray)

			newXs, newYs = estimateAllTranslation(startXs, startYs, pre_gray, cur_gray)
			Xs, Ys, newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)
			
			debug_draw(pre_gray,startXs,startYs,"pre+startX+startY")
			debug_draw(cur_gray,Xs,Ys,"cur+startX+startY")

			bbox = newbbox
			break

  		if cv2.waitKey(10) == 27:  # exit if Escape is hit
			break
	
if __name__ == "__main__":
	main()