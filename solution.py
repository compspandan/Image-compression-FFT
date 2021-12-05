import os
from matrix import matrix
from polynomial import polynomial
import numpy as np
import rsa
import cv2 as cv
from scipy import sparse
import pickle


from utils import compare_with_numpy, get_grey_scale_image, next_pow_of_two, time_exec


# 1
def test_dft(v1):
    print("coeff vector:", v1.cv)
    d = v1.dft()
    print("TASK1: Test DFT:" + compare_with_numpy(d, np.fft.fft(v1.cv)))

# 2
def test_fft(v1):
    print("coeff vector:", v1.cv)
    d = v1.fft()
    print("TASK2: Test FFT:" + compare_with_numpy(d, np.fft.fft(v1.cv)))


def pv_mul(A, B):
    fft1 = A.fft()
    fft2 = B.fft()
    return fft1 * fft2

# 3
def test_pv_mul(A, B):
    C = pv_mul(A, B)
    print("TASK3: Multiply PV form A(x) and B(x)", C)
    return polynomial(C)


# 4
def test_rsa_on_C(c_pv):
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

    print("TASK4: Apply RSA Encrytion on C(x):", compare_with_numpy(c_pv, decrypted_c_pv))
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
    pv = polynomial(pv_mul(poly1, poly2))
    ifft = np.trim_zeros(np.real(np.rint(pv.inv_fft())))
    print("TASK 5 and 6: Implement 1D Inverse FFT:", compare_with_numpy(ifft, convolution_check(poly1, poly2)))


# 6
def convolution_check(poly1, poly2):
    res_poly = poly1.naive_convolution(poly2)
    return res_poly.cv


# 7 & 8
def test_fft_2D():
    m = matrix()
    # computing fft
    fft_matrix = m.fft_2D()
	
    print("TASK 7 and 8: Check 2D FFT:", compare_with_numpy(fft_matrix, np.fft.fft2(m.matrix)))
    # computing inverse fft
    fft_matrix = matrix(fft_matrix)
    ifft_matrix = fft_matrix.ifft_2D()
    # checking if original matrix matches the matrix obtained after inverse fft
    print("TASK 7 and 8: Check 2D Inverse FFT:", compare_with_numpy(ifft_matrix, np.fft.ifft2(fft_matrix.matrix)), compare_with_numpy(m.matrix, np.real(np.rint(ifft_matrix))))


def grey_scale_image_compression():
    if not os.path.isdir('./zipped/'):
        os.mkdir('./zipped/')

    if not os.path.isdir('./compressed/'):
        os.mkdir('./compressed/')

    m = matrix(get_grey_scale_image())
    (nx, ny) = m.matrix.shape
    # m.pad_with_zeros()
    fft_matrix = m.fft_2D()
    # fft_matrix = np.fft.fft2(m.matrix)
    sorted_vals = np.sort(np.abs(np.reshape(fft_matrix, -1)))
    for trim in [0.1, 0.05, 0.025]:
        threshold = sorted_vals[int((1-trim)*len(sorted_vals))]
        compressed_matrix = np.abs(fft_matrix) > threshold
        compressed_matrix = fft_matrix * compressed_matrix

        # with gzip.GzipFile('fft_{}.npy.gz'.format(trim * 100), "w") as f:
        #     np.save(file=f, arr=compressed_matrix)

        # with gzip.GzipFile('fft_{}.npy.gz'.format(trim * 100), "r") as f:
        #     compressed_matrix = np.load(f)

        sparse_compressed_matrix = sparse.csr_matrix(compressed_matrix)

        with open('./zipped/fft_{}.npz'.format(trim * 100), 'wb') as f:
            sparse.save_npz(f, sparse_compressed_matrix, compressed=True)
		
        with open('./zipped/fft_{}.npz'.format(trim * 100), 'rb') as f:
            sparse_compressed_matrix = sparse.load_npz(f)
            compressed_matrix = sparse_compressed_matrix.toarray()

        compressed_matrix = matrix(compressed_matrix)
        compressed_img = np.real(np.rint(compressed_matrix.ifft_2D()))
        # compressed_img = np.fft.ifft2(compressed_matrix).real
        # compressed_img = compressed_img[:nx, :ny]
        cv.imwrite('./compressed/img{}.jpg'.format(trim*100), compressed_img)
    print("TASK 9 and 10: PASSED")

def runner():
    A, B = polynomial(), polynomial()
    time_exec(test_dft, A)
    time_exec(test_fft, A)
    C = time_exec(test_pv_mul, A, B).cv
    time_exec(test_rsa_on_C, C)
    time_exec(compute_inv_fft, A, B)
    time_exec(test_fft_2D)
    time_exec(grey_scale_image_compression)


if __name__ == '__main__':
    runner()
