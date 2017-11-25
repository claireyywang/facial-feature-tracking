'''
  File name: faceTracking.py
  Author: 
  Date created:
'''

'''
  File clarification:
    Generate a video with tracking features and bounding box for face regions
    - Input rawVideo: the video contains one or more faces in string format
    - Output trackedVideo: the generated video with tracked features and bounding box for face regions
'''

from optical_flow_functions import *
from basic_packages import *
from helper import *


def faceTracking(rawVideo):
	# Setup video capture
	vidcap = cv2.VideoCapture(rawVideo)

	# the previous img
	pre_img = None

	# the feature_x and feature_y in previous img  
	startXs = None 
	startYs = None

	# Read first image
	success,cur_img = vidcap.read()

	# first frame
	cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
	cur_gray = np.asarray(cur_gray)
  	bbox = detectFace(cur_gray)
	startXs, startYs = getFeatures(cur_gray,bbox)

	# init video object
	height , width , layers =  cur_img.shape
	fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
	video = cv2.VideoWriter('../Output_Video/track_video.avi',fourcc,20,(width,height),True)
	
	# write the image to video object
	cur_img_boxes = draw_boxes(cur_img,bbox) 
	video.write(cur_img_boxes)

	# update
	pre_img = cur_img

	# count i frame
	count = 0
	while (vidcap.isOpened()):
		[success, cur_img] = vidcap.read()

		if(not success):
			break
		
		# change the previous img into gray 
		pre_gray = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
		pre_gray = np.asarray(pre_gray)
			

		# change the currnet img into gray 
  		cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
		cur_gray = np.asarray(cur_gray)

		newXs, newYs = estimateAllTranslation(startXs, startYs, pre_gray, cur_gray)
		Xs, Ys, newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)

		# determine whether redetect or not 
		feature_per_box = np.array([])
		for i in xrange(len(Xs)):
			feature_per_box = np.append(feature_per_box,len(Xs[i]))

		# pdb.set_trace()
		# the box is too big or the feature is fewer than 10 
		if sum(feature_per_box < 10) > 0: 
			newbbox = detectFace(cur_gray)
			Xs, Ys = getFeatures(cur_gray,newbbox)

		# Update images and bbox
		pre_img = cur_img
		startXs = Xs
		startYs = Ys
		bbox = newbbox

		# draw the box on image
		cur_img_boxes = draw_boxes(cur_img,bbox) 
		video.write(cur_img_boxes)

		# for tracking
		print('{} frame finish').format(count)
		count += 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# release
	cv2.destroyAllWindows()
	video.release()
	vidcap.release()
	
	trackedVideo = '../Output_Video/track_video.avi'
	return trackedVideo