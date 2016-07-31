#! /usr/bin/env python
# encoding=utf-8

#--------------------------------------------------------------
# shuffle datasets.
# author: zhouming402@163.com
# date: 2016-07-23
#--------------------------------------------------------------

import os, sys
import os.path as osp
import argparse
import random

DEBUG = True

def parse_args():
	parser = argparse.ArgumentParser(description='shuffle samples')

	parser.add_argument('--data', dest='data_path',
			help='eye datasets path', default=None, type=str)
	parser.add_argument('--stat', dest='stat_path',
			help='stat save path', default=None, type=str)
		
	args = parser.parse_args()

	return args

if __name__ == '__main__':
	
	args = parse_args()

	with open(args.data_path, 'r') as f:
		data = [line for line in f]
		random.shuffle(data)

	if args.stat_path is not None:
		with open(args.stat_path, 'w') as f:
			[f.write(line) for line in data]
	else:
		with open(args.data_path, 'w') as f:
			[f.write(line) for line in data]


