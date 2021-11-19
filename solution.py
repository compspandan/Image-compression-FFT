from polynomial import polynomial
import numpy as np


def test_dft_fft_1n2():
    v1 = polynomial()
    print(v1.cv)
    d = v1.dft()
    print(np.allclose(d, np.fft.fft(v1.cv)))
    d = v1.fft()
    print(np.allclose(d, np.fft.fft(v1.cv)))


def pv_mul(fft1, fft2):
    l1 = len(fft1)
    l2 = len(fft2)
    l3 = l1 + l2


def test_pv_mul_3():
    poly1 = polynomial()
    poly2 = polynomial()
    fft1 = poly1.fft()
    fft2 = poly2.fft()
    pv_mul(fft1, fft2)


def convolution_check():
    p4 = polynomial()
    p3 = polynomial(deg_bound=3)
    p7 = p4.naive_convolution(p3)
    print(p3.cv, p4.cv, p7.cv, end='\n\n')
