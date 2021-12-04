from numpy import fft
from matrix import matrix
from polynomial import polynomial
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from utils import next_pow_of_two

# 1 and 2
def test_dft_fft():
    v1 = polynomial()
    print("coeff vec:",v1.cv)
    d = v1.dft()
    print(np.allclose(d, np.fft.fft(v1.cv)))
    d = v1.fft()
    print(np.allclose(d, np.fft.fft(v1.cv)))

# 3
def test_pv_mul(poly1,poly2):
	fft1 = poly1.fft()
	fft2 = poly2.fft()
	return polynomial(fft1*fft2)

# 5
def compute_inv_fft():
	poly1 = polynomial()
	poly2 = polynomial()
	deg_bound = next_pow_of_two(poly1.deg_bound + poly2.deg_bound - 1)
	# padding cv of poly1 and poly2 to next highest power of 2 for IFFT
	poly1.cv = np.pad(poly1.cv,(0,deg_bound-poly1.deg_bound))
	poly2.cv = np.pad(poly2.cv,(0,deg_bound-poly2.deg_bound))

	# print('poly1',poly1.cv)
	# print('poly2',poly2.cv)
	pv = test_pv_mul(poly1,poly2)
	ifft = np.trim_zeros(np.real(np.rint(pv.inv_fft())))
	print(np.allclose(ifft, convolution_check(poly1,poly2)))

# 6
def convolution_check(poly1,poly2):
    res_poly = poly1.naive_convolution(poly2)
    return res_poly.cv

# 7 & 8
def test_fft_2D():
	m = matrix()
	# computing fft
	fft_matrix = m.fft_2D()
	print(np.allclose(fft_matrix, np.fft.fft2(m.matrix)))
	# computing inverse fft
	fft_matrix = matrix(fft_matrix)
	ifft_matrix = fft_matrix.ifft_2D()
	print(np.allclose(ifft_matrix, np.fft.ifft2(fft_matrix.matrix)))
	# checking if original matrix matches the matrix obtained after inverse fft
	print(np.allclose(m.matrix,np.real(np.rint(ifft_matrix))))

def get_grey_scale_matrix():
	img = cv.imread('bw_rose.jpg')
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	return matrix(img)

def grey_scale_image_compression():
	m = get_grey_scale_matrix()
	(nx,ny) = m.matrix.shape
	m.pad_with_zeros()
	fft_matrix = m.fft_2D()
	# fft_matrix = np.fft.fft2(m.matrix)
	sorted_vals = np.sort(np.abs(np.reshape(fft_matrix,-1)))
	for trim in [0.1,0.05,0.025]:
		threshold = sorted_vals[int((1-trim)*len(sorted_vals))]
		compressed_matrix = np.abs(fft_matrix)>threshold
		compressed_matrix = fft_matrix * compressed_matrix
		cv.imwrite('fft_image'+str(trim*100)+'.jpg',np.real(compressed_matrix))
		compressed_matrix = matrix(compressed_matrix)
		compressed_img = np.real(np.rint(compressed_matrix.ifft_2D()))
		# compressed_img = np.fft.ifft2(compressed_matrix).real
		(m,n) = fft_matrix.shape
		compressed_img = compressed_img[:nx,:ny]
		cv.imwrite('comp_img'+str(trim*100)+'.jpg',compressed_img)

grey_scale_image_compression()