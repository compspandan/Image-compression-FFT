from random import randint
import cv2
import numpy as np


class polynomial:
	max_coeff = 100

	def __init__(self, cv=None):
		if cv is None:
			# self.deg_bound = pow(2,randint(2,11))
			self.deg_bound = 4
			self.cv = np.random.randint(1, self.max_coeff, self.deg_bound)
		else:
			self.deg_bound = len(cv)
			self.cv = np.array(cv,dtype=complex)

	def dft(self, cv=None):
		if cv is None:
			cv = self.cv
		cv, n = np.array(cv, dtype=complex), len(cv)
		arr = np.array([number for number in range(n)])
		values = np.matmul(arr.reshape((n, 1)), arr.reshape(1, n))
		base = -2j * (np.pi / n) * values
		vander = np.exp(base)
		pv = vander @ cv
		return pv 

	def fft(self, cv=None):
		if cv is None:
			cv = self.cv
		cv, n = np.asarray(cv, dtype=complex), len(cv)
		if n & 1:
			raise ValueError("cv is not a power of 2")
		elif n <= 2:
			return self.dft(cv)
		w_n, w = np.exp(-2j * np.pi / n), 1
		y_0, y_1 = self.fft(cv[::2]), self.fft(cv[1::2])
		y = np.zeros(n, dtype=complex)
		for k in range(n // 2):
			y[k] = y_0[k] + w * y_1[k]
			y[k + n//2] = y_0[k] - w * y_1[k]
			w = w * w_n
		return y

	
	def inv_dft(self, cv=None):
		if cv is None:
			cv = self.cv
		cv, n = np.asarray(cv, dtype=complex), len(cv)
		arr = np.array([number for number in range(n)])
		values = -1 * np.matmul(arr.reshape((n, 1)), arr.reshape(1, n))
		base = -2j * (np.pi / n) * values
		vander = np.exp(base)
		if np.array_equal(cv, self.cv):
			return 1 / n * np.dot(vander, cv)
		return vander @ cv

	def inv_fft(self, cv=None):
		if cv is None:
			cv = self.cv
		cv, n = np.asarray(cv, dtype=complex), len(cv)
		if n & 1:
			raise ValueError("cv is not a power of 2")
		elif n <= 2:
			return self.inv_dft(cv)
		w_n, w = np.exp(2j * np.pi / n), 1
		y_0, y_1 = self.inv_fft(cv[::2]), self.inv_fft(cv[1::2])
		y = np.zeros(n, dtype=complex)
		for k in range(n // 2):
			y[k] = y_0[k] + w * y_1[k]
			y[k + n//2] = y_0[k] - w * y_1[k]
			w = w * w_n
		return y / n if np.array_equal(cv, self.cv) else y
	
	def naive_convolution(self, B):
		A = self
		n, m = A.deg_bound, B.deg_bound
		p = n + m - 1
		C = polynomial(np.zeros(p))
		for i in range(n):
			for j in range(m):
				C.cv[i + j] += A.cv[i] * B.cv[j]
		return C
