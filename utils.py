import numpy as np
from timeit import default_timer as timer
import cv2 as cv


def next_pow_of_two(n):
	# incase n itself is power of 2
	n = n - 1
	while n & n - 1:
		n = n & n - 1 
	return n << 1

def time_exec(func, *args):
	start = timer()
	res = func(*args)
	end = timer()
	print('time taken: [{:5f}s]\n'.format(end - start))
	return res

def compare_with_numpy(x,y):
	if np.allclose(x,y):
		return 'PASSED'
	else:
		return 'FAILED'

def get_grey_scale_image():
    img = cv.imread('bw_rose.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img
