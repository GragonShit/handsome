#! /usr/bin/env python

#---------------------------------------------------------------
# test opencv face detection
# author: zhouming402@163.com
# date: 2016-08-07
#---------------------------------------------------------------


import numpy as np
import cv2

if __name__ == '__main__':
	frontal_face_cascade = cv2.CascadeClassifier('../../model/haarcascades/haarcascade_frontalface_alt.xml')
	profile_face_cascade = cv2.CascadeClassifier('../../model/haarcascades/haarcascade_profileface.xml')
	
	cap = cv2.VideoCapture(0)

	while True:
		
		ret, frame = cap.read()
		
		frame = cv2.resize(frame, (frame.shape[1]/2,frame.shape[0]/2))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		frontal_faces = frontal_face_cascade.detectMultiScale(gray, 1.3, 5)
		profile_faces = profile_face_cascade.detectMultiScale(gray, 1.3, 5)

		for (x,y,w,h) in frontal_faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

		for (x,y,w,h) in profile_faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

		cv2.imshow('frame', frame)
		cv2.waitKey(1)
