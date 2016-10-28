#! /usr/bin/env python
# encoding=utf-8

#--------------------------------------------------------------
# extract tongue's patchs from datasets.
# author: zhouming402@163.com
# date: 2016-10-26
#--------------------------------------------------------------

import os, sys
import os.path as osp
import dlib
import argparse
import cv2
from util import tongue_region

DEBUG = True

def parse_args():
	parser = argparse.ArgumentParser(description='make sample')

	parser.add_argument('--data', dest='data_path',
			help='face datasets path', default=None, type=str)
	parser.add_argument('--face', dest='face_shape',
			help='face shape predictor path', default=None, type=str)
	parser.add_argument('--stat', dest='stat_path',
			help='stat save path', default=None, type=str)
		
	args = parser.parse_args()

	return args

def save_tongue(img, rect, path, jpg):
	cv2.imwrite(osp.join(path,jpg+'_tongue.jpg'), 
			img[rect.top():rect.bottom()+1,rect.left():rect.right()+1])

if __name__ == '__main__':
	
	args = parse_args()

	face_det = dlib.get_frontal_face_detector()
	face_shaper = dlib.shape_predictor(args.face_shape)

	if not osp.exists(args.stat_path):
		os.makedirs(args.stat_path)

	for dir in os.listdir(args.data_path):
		path = osp.join(args.data_path, dir)
		spath = osp.join(args.stat_path, dir)
		if not osp.exists(spath):
			os.makedirs(spath)
		if DEBUG:
			print path
		if osp.isdir(path):
			for jpg in os.listdir(path):
				if DEBUG:
					print jpg
				if osp.splitext(jpg)[1] == '.jpg':
					img = cv2.imread(osp.join(path, jpg))
					dets = face_det(img, 0)
					for k,d in enumerate(dets):
						if DEBUG:
							print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
										k, d.left(), d.top(), d.right(), d.bottom()))
						face = face_shaper(img, d)
						rect = tongue_region(face)
						print rect
						save_tongue(img, rect, spath, osp.splitext(jpg)[0]+'_'+str(k))


