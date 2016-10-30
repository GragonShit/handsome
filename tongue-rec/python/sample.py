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
from util import tongue_region, lip_region

DEBUG = True

def parse_args():
	parser = argparse.ArgumentParser(description='make sample')

	parser.add_argument('--data', dest='data_path',
			help='face datasets path', default=None, type=str)
	parser.add_argument('--mode', dest='mode',
			help='tongue region or lip region', default="lip", type=str)
	parser.add_argument('--face', dest='face_shape',
			help='face shape predictor path', default=None, type=str)
	parser.add_argument('--stat', dest='stat_path',
			help='stat save path', default=None, type=str)
	parser.add_argument('--aug', dest='augment',
			help='if augment', default=False, type=bool)
		
	args = parser.parse_args()

	return args

def aug(img, rect, path, jpg):
	wstep = rect.width()/2
	hstep = rect.height()/2
	tongue = img[rect.top()-hstep:rect.bottom()+1+hstep,\
		  rect.left()-wstep:rect.right()+1+wstep].copy()
	index = 0
	# angle
	for angle in range(-30,40,10):
		M = cv2.getRotationMatrix2D((tongue.shape[1]/2,tongue.shape[0]/2), angle, 1.0)
		newtongue = cv2.warpAffine(tongue, M, (tongue.shape[1],tongue.shape[0]))
		for shift in range(-10,15,5):
			spath = osp.join(path,jpg+'_{}.jpg'.format(index))
			cv2.imwrite(spath, 
					newtongue[int(rect.height()*shift/100.0+hstep):int(rect.height()*shift/100.0+hstep*3),
					int(rect.width()*shift/100.0+wstep):int(rect.width()*shift/100.0+wstep*3)])
			index += 1

def save(img, rect, path, jpg):
	cv2.imwrite(osp.join(path,jpg+'.jpg'), 
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
						if args.mode == 'tongue':
							rect = tongue_region(face)
						elif args.mode == 'lip':
							rect = lip_region(face)
						print rect
						if args.augment:
							aug(img, rect, spath, osp.splitext(jpg)[0]+'_'+str(k))
						else:
							save(img, rect, spath, osp.splitext(jpg)[0]+'_'+str(k))
						break; # one image one face


