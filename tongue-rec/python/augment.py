#! /usr/bin/env python
# encoding=utf-8

#--------------------------------------------------------------
# augment eyes' patchs from datasets.
# author: zhouming402@163.com
# date: 2016-10-26
#--------------------------------------------------------------

import os, sys
import os.path as osp
import dlib
import argparse
import cv2
import numpy as np
from util import tongue_region

DEBUG = True

def parse_args():
	parser = argparse.ArgumentParser(description='data augment')

	parser.add_argument('--list', dest='list_path',
			help='face datasets path', default=None, type=str)
	parser.add_argument('--source', dest='source_path',
			help='source datasets path', default=None, type=str)
	parser.add_argument('--face', dest='face_shape',
			help='face shape predictor path', default=None, type=str)
	parser.add_argument('--stat', dest='stat_path',
			help='stat save path', default=None, type=str)
		
	args = parser.parse_args()

	return args

def aug_eyes(img, rect, path, f, name, tag):
	# f = open(osp.join(path, jpg+'.txt'), 'w')
	wstep = rect.width()/2
	hstep = rect.height()/2
	eye = img[rect.top()-hstep:rect.bottom()+1+hstep,\
		  rect.left()-wstep:rect.right()+1+wstep].copy()
	index = 0
	# angle
	for angle in range(-30,40,10):
		M = cv2.getRotationMatrix2D((eye.shape[1]/2,eye.shape[0]/2), angle, 1.0)
		neweye = cv2.warpAffine(eye, M, (eye.shape[1],eye.shape[0]))
		for shift in range(-10,15,5):
			spath = osp.join(path,name.replace('.jpg', '_{}.jpg'.format(index)))
			if not osp.exists(osp.dirname(spath)):
				os.makedirs(spath)
			cv2.imwrite(osp.join(path,name.replace('.jpg', '_{}.jpg'.format(index))), 
					neweye[int(rect.height()*shift/100.0+hstep):int(rect.height()*shift/100.0+hstep*3),
					int(rect.width()*shift/100.0+wstep):int(rect.width()*shift/100.0+wstep*3)])
			ftxt.write('{}\t{}\n'.format(name.replace('.jpg', '_{}.jpg'.format(index)), tag))
			index += 1

if __name__ == '__main__':
	
	args = parse_args()

	face_det = dlib.get_frontal_face_detector()
	face_shaper = dlib.shape_predictor(args.face_shape)

	if not osp.exists(args.stat_path):
		os.makedirs(args.stat_path)

	ftxt = open(osp.join(args.stat_path, 'list_train.txt'), 'w')
	
	for line in open(args.list_path, 'r'):
		infos = line.strip().split('\t')
		name,_,tag = infos[0].rpartition('_')
		tag = tag.split('.')[0]
		path = osp.join(args.source_path, name.rpartition('_')[0]+'.jpg')

		if DEBUG:
			print name,tag
		img = cv2.imread(path)
		dets = face_det(img, 0)

		face = face_shaper(img, dets[0])
		rect = tongue_region(face)
		# eyel = eye_shaper(img, rectl)
		# eyer = eye_shaper(img, rectr)
		# save_eyes(img, rectl, rectr, spath, osp.splitext(jpg)[0]+'_'+str(k), face, eyel, eyer)
		if tag == 'eyel':
			aug_eyes(img, rectl, args.stat_path, ftxt, infos[0], infos[1])
		elif tag == 'eyer':
			aug_eyes(img, rectr, args.stat_path, ftxt, infos[0], infos[1])
	ftxt.close()

