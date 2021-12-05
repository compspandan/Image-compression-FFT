import os
import cv2 as cv
import numpy as np
from scipy import sparse

import rsa
from matrix import matrix
from polynomial import polynomial
from utils import compare_with_numpy, get_grey_scale_image, next_pow_of_two, time_exec


# 1
def test_dft(v1):
    print("coeff vector for DFT:", v1.cv)
    return v1.dft()

# 2
def test_fft(v1):
    print("coeff vector for FFT:", v1.cv)
    return v1.fft()


def pv_mul(A, B):
    fft1 = A.fft()
    fft2 = B.fft()
    return fft1 * fft2

# 3
def test_pv_mul(A, B):
    C = pv_mul(A, B)
    print("Multiply PV form A(x) and B(x)", C)
    return polynomial(C)


# 4
def test_rsa_on_C(c_pv):
    c_pv_str = "_".join(list(map(str, c_pv)))
    c_pv_num_blocks = []
    c_pv_num_curr = 0
    CHAR_SIZE, BLOCK_SIZE = 1000, 100
    for i, c in enumerate(c_pv_str):
        if i % BLOCK_SIZE == 0 and i != 0:
            c_pv_num_blocks.append(c_pv_num_curr)
            c_pv_num_curr = 0
        c_pv_num_curr = c_pv_num_curr * CHAR_SIZE + ord(c)
    c_pv_num_blocks.append(c_pv_num_curr)

    public_key, secret_key = rsa.create(512)
    cipher = list(map(lambda x: rsa.encrypt(x, public_key), c_pv_num_blocks))
    decrypted_c_pv_num_blocks = list(
        map(lambda x: rsa.decrypt(x, secret_key), cipher))

    decrypted_c_pv_str = ""
    for block in decrypted_c_pv_num_blocks:
        decrypted_c_pv_block_str = ""
        for _ in range(BLOCK_SIZE):
            decrypted_c_pv_block_str = chr(
                block % CHAR_SIZE) + decrypted_c_pv_block_str
            block //= CHAR_SIZE
        decrypted_c_pv_str = decrypted_c_pv_str + decrypted_c_pv_block_str
    decrypted_c_pv = list(map(lambda x: complex(
        x.replace('\x00', '')), decrypted_c_pv_str.split("_")))

    print("Apply RSA Encrytion on C(x):", compare_with_numpy(c_pv, decrypted_c_pv))
    if np.allclose(c_pv, decrypted_c_pv):
        print("RSA Encrpytion Decrytion on C(X) successful")
        print("C(x):", c_pv)
        print("C(x) after RSA encryption and decryption:", decrypted_c_pv)
    else:
        raise Exception("RSA Encrpytion Failed")


# 5
def compute_inv_fft(poly1, poly2):
    deg_bound = next_pow_of_two(poly1.deg_bound + poly2.deg_bound - 1)
    # padding cv of poly1 and poly2 to next highest power of 2 for IFFT
    poly1.cv = np.pad(poly1.cv, (0, deg_bound-poly1.deg_bound))
    poly2.cv = np.pad(poly2.cv, (0, deg_bound-poly2.deg_bound))
    poly1_fft = np.fft.fft(poly1.cv)
    poly2_fft = np.fft.fft(poly2.cv)
    pv = polynomial(poly1_fft * poly2_fft)
    ifft = np.trim_zeros(np.real(np.rint(pv.inv_fft())))
    return ifft, pv


# 6
def convolution_check(poly1, poly2):
    res_poly = poly1.naive_convolution(poly2)
    return res_poly.cv


# 7 & 8
def test_fft_2D(m):

    # computing fft
    fft_matrix = m.fft_2D()
	
    # computing inverse fft
    fft_matrix = matrix(fft_matrix)
    ifft_matrix = fft_matrix.ifft_2D()
    # checking if original matrix matches the matrix obtained after inverse fft
    return (fft_matrix.matrix, ifft_matrix)


def grey_scale_image_compression():
    if not os.path.isdir('../zipped/'):
        os.mkdir('../zipped/')

    if not os.path.isdir('../compressed/'):
        os.mkdir('../compressed/')

    m = matrix(get_grey_scale_image())

    # uncomment to pad
    # (nx, ny) = m.matrix.shape
    # m.pad_with_zeros()

    fft_matrix = m.fft_2D()
    sorted_vals = np.sort(np.abs(np.reshape(fft_matrix, -1)))
    for trim in [0.1, 0.05, 0.025]:
        threshold = sorted_vals[int((1-trim)*len(sorted_vals))]
        compressed_matrix = np.abs(fft_matrix) > threshold
        compressed_matrix = fft_matrix * compressed_matrix

        sparse_compressed_matrix = sparse.csr_matrix(compressed_matrix)

        with open('../zipped/fft_{}.npz'.format(trim * 100), 'wb') as f:
            sparse.save_npz(f, sparse_compressed_matrix, compressed=True)
		
        with open('../zipped/fft_{}.npz'.format(trim * 100), 'rb') as f:
            sparse_compressed_matrix = sparse.load_npz(f)
            compressed_matrix = sparse_compressed_matrix.toarray()

        compressed_matrix = matrix(compressed_matrix)
        compressed_img = np.real(np.rint(compressed_matrix.ifft_2D()))

        # uncomment to remove padding
        # compressed_img = compressed_img[:nx, :ny]
        
        cv.imwrite('../compressed/img{}.jpg'.format(trim*100), compressed_img)
    print("PASSED")

def runner():
    A, B = polynomial(), polynomial()
    print("TASK 1:")
    pv = time_exec('Exec Time DFT',test_dft, A)
    test_pv = time_exec('Exec Time Numpy FFT',np.fft.fft, A.cv)
    print("Test DFT:" + compare_with_numpy(pv, test_pv))
    print()

    print("TASK 2:") 
    pv = time_exec('Time FFT',test_fft, A)
    print("Test FFT:" + compare_with_numpy(pv, test_pv))
    test_pv = time_exec('Exec Time Numpy FFT',np.fft.fft, A.cv)
    print()

    print("TASK 3:")
    C = time_exec('Exec Time PV multiplication',test_pv_mul, A, B).cv
    print()

    print("TASK 4:")
    time_exec('Exec Time RSA',test_rsa_on_C, C)
    print()

    print("TASK 5 and 6:")
    ifft, pv = time_exec('Exec Time Inverse FFT', compute_inv_fft, A, B)
    np_ifft = time_exec("Exec Time Numpy IFFT:", np.fft.ifft, pv.cv)
    mul = time_exec('Exec time elementary polynomial multiplication',convolution_check,A, B)

    print("Implement 1D Inverse FFT:", compare_with_numpy(ifft, np.trim_zeros(np.real(np.rint(np_ifft)))))
    print("Implement 1D Inverse FFT:", compare_with_numpy(ifft, mul))
    print()

    print("TASK 7 AND 8:")
    m = matrix()
    (fft_matrix,ifft_matrix) = time_exec('Exec Time 2D FFT & Inverse FFT',test_fft_2D,m)
    np_fft_matrix = time_exec('Exec Time Numpy 2D FFT',np.fft.fft2,m.matrix)
    print("Check 2D FFT:", compare_with_numpy(fft_matrix, np_fft_matrix))
    np_ifft_matrix = time_exec('Exec Time Numpy 2D IFFT',np.fft.ifft2, fft_matrix)
    print("Check 2D Inverse FFT:", compare_with_numpy(ifft_matrix, np_ifft_matrix))
    print("Check if matrices match after FFT and IFFT:", compare_with_numpy(m.matrix, np.real(np.rint(ifft_matrix))))
    print()

    # print("TASK 9 AND 10:")
    # time_exec("Compress Grayscale Images via FFT:", grey_scale_image_compression)
    # print()


if __name__ == '__main__':
    runner()
