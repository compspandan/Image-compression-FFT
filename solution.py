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
	print('fft1',fft1)
	print('fft2',fft2)
	return polynomial(fft1*fft2)

def compute_inv_fft():
	poly1 = polynomial([1,1,1,0,0,0,0,0])
	poly2 = polynomial([1,1,1,0,0,0,0,0])
	print('poly1',poly1.cv,'poly2',poly2.cv)
	pv = test_pv_mul(poly1,poly2)
	print('pv',pv.cv)
	# print('ifft',pv.inv_fft())
	# print('dft',pv.inv_dft())
	print('np fft',np.fft.ifft(pv.cv))
	# print(np.allclose(pv.inv_fft(),np.fft.ifft(pv.cv)))
	convolution_check(poly1,poly2)

# 6
def convolution_check(poly1,poly2):
    res_poly = poly1.naive_convolution(poly2)
    print(res_poly.cv)

compute_inv_fft()