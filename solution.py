from polynomial import polynomial
import numpy as np


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

def next_pow_of_two(n):
	# incase n itself is power of 2
	n = n - 1
	while n & n - 1:
		n = n & n - 1 
	return n << 1

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

compute_inv_fft()