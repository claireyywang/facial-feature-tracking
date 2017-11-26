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


def drawBox(img, bbox):
  imgwbox = img.copy()
  for box in bbox:
    l_t_pt = (box[0])[::-1]
    r_b_pt = box[3][::-1]
    cv2.rectangle(imgwbox,tuple(l_t_pt),tuple(r_b_pt),(0,255,0),2)
  return imgwbox