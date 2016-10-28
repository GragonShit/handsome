#! /usr/bin/env python
# encoding=utf-8

#--------------------------------------------------------------
# util functions
# author: zhouming402@163.com
# date: 2016-10-26
#--------------------------------------------------------------

import os
import cv2
import numpy as np
import dlib

def tongue_region(shape):
	points = np.ndarray((12,1,2), dtype=np.float32)
	
	for i in range(48,60):
		points[i-48,0] = (shape.part(i).x, shape.part(i).y)

	rect = cv2.boundingRect(points)
	xl = rect[0]
	yl = rect[1]
	xr = rect[0]+rect[2]
	yr = rect[1]+rect[3]

	'''
	rect = dlib.rectangle(int(xl-w),int(yl-h),
			int(xl+w),int(yl+h))
	'''
	rect = dlib.rectangle(int(xl),int(yl),int(xr),int(yr))

	return rect


if __name__ == '__main__':
	
	pass
