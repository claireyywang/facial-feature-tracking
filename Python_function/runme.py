'''
  File clarification:
	Initializing the face tracking function
'''


from faceTracking import *

def main():
	rawVideo = '../Datasets/Easy/MarquesBrownlee.mp4'
	trackedVideo = faceTracking(rawVideo) 


if __name__ == "__main__":
	main()