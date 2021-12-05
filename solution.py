from matrix import matrix
from polynomial import polynomial
import numpy as np
import rsa


# 1 and 2
def test_dft_fft():
    v1 = polynomial()
    print("coeff vec:", v1.cv)
    d = v1.dft()
    print(np.allclose(d, np.fft.fft(v1.cv)))
    d = v1.fft()
    print(np.allclose(d, np.fft.fft(v1.cv)))


# 3
def test_pv_mul(poly1, poly2):
    fft1 = poly1.fft()
    fft2 = poly2.fft()
    return polynomial(fft1*fft2)


# 4
def test_rsa_on_C(a, b):
	a_pv = a.fft()
	b_pv = b.fft()
	c_pv = a_pv * b_pv
	
	c_pv_str = "_".join(list(map(str, c_pv)))
	c_pv_num_blocks = []
	c_pv_num_curr = 0
	CHAR_SIZE, BLOCK_SIZE = 1000, 2
	for i, c in enumerate(c_pv_str):
		if i % BLOCK_SIZE == 0 and i != 0:
			c_pv_num_blocks.append(c_pv_num_curr)
			c_pv_num_curr = 0
		c_pv_num_curr = c_pv_num_curr * CHAR_SIZE + ord(c)
	c_pv_num_blocks.append(c_pv_num_curr)

	public_key, secret_key = rsa.create(512)
	cipher = list(map(lambda x: rsa.encrypt(x, public_key), c_pv_num_blocks))
	decrypted_c_pv_num_blocks = list(map(lambda x: rsa.decrypt(x, secret_key), cipher))

	decrypted_c_pv_str = ""
	for block in decrypted_c_pv_num_blocks:
		decrypted_c_pv_block_str = ""
		for _ in range(BLOCK_SIZE):
			decrypted_c_pv_block_str = chr(block % CHAR_SIZE) + decrypted_c_pv_block_str
			block //= CHAR_SIZE
		decrypted_c_pv_str = decrypted_c_pv_str + decrypted_c_pv_block_str
	decrypted_c_pv = list(map(lambda x: complex(x.replace('\x00', '')), decrypted_c_pv_str.split("_")))

	if np.allclose(c_pv, decrypted_c_pv):
		print("RSA Encrpytion Decrytion on C(X) successfull")
		print("C(x):", c_pv)
		print("C(x) after RSA encryption and decryption:", decrypted_c_pv)
	else:
		raise Exception("RSA Encrpytion Failed")


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
    poly1.cv = np.pad(poly1.cv, (0, deg_bound-poly1.deg_bound))
    poly2.cv = np.pad(poly2.cv, (0, deg_bound-poly2.deg_bound))

    # print('poly1',poly1.cv)
    # print('poly2',poly2.cv)
    pv = test_pv_mul(poly1, poly2)
    ifft = np.trim_zeros(np.real(np.rint(pv.inv_fft())))
    print(np.allclose(ifft, convolution_check(poly1, poly2)))


# 6
def convolution_check(poly1, poly2):
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
    print(np.allclose(m.matrix, np.real(np.rint(ifft_matrix))))


if __name__ == "__main__":
    poly1, poly2 = polynomial(), polynomial()
    test_rsa_on_C(poly1, poly2)
