
import os
import os.path as osp
import sys

import cv2
import numpy as np

if __name__ == '__main__':
	
	for jpg in os.listdir(sys.argv[1]):
		print jpg
		im = cv2.imread(osp.join(sys.argv[1],jpg))
		im = np.transpose(im, (1,0,2))
		cv2.imwrite(osp.join(sys.argv[2],jpg), im)
