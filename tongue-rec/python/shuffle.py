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
	parser.add_argument('--stat_train', dest='stat_train',
			help='stat save path', default=None, type=str)
	parser.add_argument('--stat_test', dest='stat_test',
			help='stat save path', default=None, type=str)
		
	args = parser.parse_args()

	return args

if __name__ == '__main__':
	
	args = parse_args()

	with open(args.data_path, 'r') as f:
		data = [line for line in f]
		data_train = [j for i,j in enumerate(data) if i%10!=0]
		data_test = [j for i,j in enumerate(data) if i%10==0]
		random.shuffle(data_train)
		random.shuffle(data_test)

	with open(args.stat_train, 'w') as f:
		[f.write(line) for line in data_train]
	with open(args.stat_test, 'w') as f:
		[f.write(line) for line in data_test]


