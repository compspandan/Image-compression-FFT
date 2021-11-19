from numpy.lib.polynomial import poly
from polynomial import polynomial
import numpy as np

def test_dft_fft_1n2():
	v1 = polynomial()
	print(v1.cv)
	d = v1.dft()
	print(np.allclose(d, np.fft.fft(v1.cv)))
	d = v1.fft()
	print(np.allclose(d, np.fft.fft(v1.cv)))

def pv_mul(fft1,fft2,poly1,poly2):
	l1 = len(fft1)
	l2 = len(fft2)
	l3 = l1 + l2
	for _ in range(l3-l1):
		fft1 = np.append(fft1,poly1.cv[0])
	for _ in range(l3-l2):
		fft2 = np.append(fft2,poly2.cv[0])
	return fft1*fft2

def test_pv_mul_3():
	poly1 = polynomial()
	poly2 = polynomial()
	fft1 = poly1.fft()
	fft2 = poly2.fft()
	pv = pv_mul(fft1,fft2,poly1,poly2)
	print(pv)