#! /usr/bin/env python
# encoding=utf-8

#--------------------------------------------------------------
# util functions
# author: zhouming402@163.com
# date: 2016-07-23
#--------------------------------------------------------------

import os
import cv2
import numpy as np
import dlib

def eye_region(shape):
	pointsl = np.ndarray((6,1,2), dtype=np.float32)
	pointsr = np.ndarray((6,1,2), dtype=np.float32)
	
	for i in range(36,42):
		pointsl[i-37,0] = (shape.part(i).x, shape.part(i).y)
	for i in range(42,48):
		pointsr[i-43,0] = (shape.part(i).x, shape.part(i).y)

	rectl = cv2.boundingRect(pointsl)
	rectr = cv2.boundingRect(pointsr)

	xl = rectl[0] + rectl[2]/2.0
	yl = rectl[1] + rectl[3]/2.0
	xr = rectr[0] + rectr[2]/2.0
	yr = rectr[1] + rectl[3]/2.0

	w = (xr - xl) * 3 / 10
	h = (xr - xl) / 10

	rectl = dlib.rectangle(int(xl-w),int(yl-h),
			int(xl+w),int(yl+h))
	rectr = dlib.rectangle(int(xr-w),int(yr-h),
			int(xr+w),int(yr+h))

	return rectl,rectr


def mouth_region(shape):
	points = np.ndarray((6,1,2), dtype=np.float32)
	
	for i in range(36,42):
		points[i-37,0] = (shape.part(i).x, shape.part(i).y)

	rect = cv2.boundingRect(points)

	'''
	x = rect[0] + rect[2]/2.0
	y = rect[1] + rect[3]/2.0

	rectl = dlib.rectangle(int(xl-w),int(yl-h),
			int(xl+w),int(yl+h))
	'''

	return rect

if __name__ == '__main__':
	
	pass
