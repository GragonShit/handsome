#! /usr/bin/env python
# encoding=utf-8

#--------------------------------------------------------------
# extract eyes' patchs from datasets.
# author: zhouming402@163.com
# date: 2016-07-23
#--------------------------------------------------------------

import os, sys
import os.path as osp
import dlib
import argparse
import cv2
from util import eye_region

DEBUG = True

def parse_args():
	parser = argparse.ArgumentParser(description='')

	parser.add_argument('--data', dest='data_path',
			help='face datasets path', default=None, type=str)
	parser.add_argument('--face', dest='face_shape',
			help='face shape predictor path', default=None, type=str)
	parser.add_argument('--eye', dest='eye_shape',
			help='eye shape predictor path', default=None, type=str)
	parser.add_argument('--stat', dest='stat_path',
			help='stat save path', default=None, type=str)
		
	args = parser.parse_args()

	return args

def save_eyes(img, rectl, rectr, path, jpg, face=None, eyel=None, eyer=None):
	# f = open(osp.join(path, jpg+'.txt'), 'w')
	if face:
		for i in range(37, 43):
			cv2.circle(img, (face.part(i).x,face.part(i).y), 4, (0,0,255))
		for i in range(43, 49):
			cv2.circle(img, (face.part(i).x,face.part(i).y), 4, (0,0,255))

	if eyel:
		for i in range(0, 12):
			cv2.circle(img, (eyel.part(i).x,eyel.part(i).y), 4, (0,255,0))
		for i in range(12, 20):
			cv2.circle(img, (eyel.part(i).x,eyel.part(i).y), 4, (255,0,0))

	if eyer:
		for i in range(0, 12):
			cv2.circle(img, (eyer.part(i).x,eyer.part(i).y), 4, (0,255,0))
		for i in range(12, 20):
			cv2.circle(img, (eyer.part(i).x,eyer.part(i).y), 4, (255,0,0))
	cv2.imwrite(osp.join(path,jpg+'_eyel.jpg'), 
			img[rectl.top():rectl.bottom()+1,rectl.left():rectl.right()+1])
	cv2.imwrite(osp.join(path,jpg+'_eyer.jpg'), 
			img[rectr.top():rectr.bottom()+1,rectr.left():rectr.right()+1])
	
	# f.close()


if __name__ == '__main__':
	
	args = parse_args()

	face_det = dlib.get_frontal_face_detector()
	face_shaper = dlib.shape_predictor(args.face_shape)
	eye_shaper = dlib.shape_predictor(args.eye_shape)

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
						rectl,rectr = eye_region(face)
						eyel = eye_shaper(img, rectl)
						eyer = eye_shaper(img, rectr)
						# save_eyes(img, rectl, rectr, spath, osp.splitext(jpg)[0]+'_'+str(k), face, eyel, eyer)
						save_eyes(img, rectl, rectr, spath, osp.splitext(jpg)[0]+'_'+str(k))


