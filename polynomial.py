from random import randint
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

	def dft(self, x=None):
		if x is None:
			x = self.cv
		x = np.asarray(x, dtype=complex)
		N = x.shape[0]
		n = np.arange(N)
		k = n.reshape((N, 1))
		M = np.exp(-2j * np.pi/N * k * n)
		return np.dot(M, x)

	def fft(self, x=None):
		if x is None:
			x = self.cv
		x = np.asarray(x, dtype=complex)
		N = x.shape[0]
		if N % 2 > 0:
			raise ValueError("must be a power of 2")
		elif N <= 2:
			return self.dft(x)
		else:
			X_even = self.fft(x[::2])
			X_odd = self.fft(x[1::2])
			terms = np.exp(-2j * np.pi/N * np.arange(N))
		return np.concatenate([X_even + terms[:int(N/2)] * X_odd, X_even + terms[int(N/2):] * X_odd])
	
	def inv_dft(self, x=None):
		if x is None:
			x = self.cv
		x = np.asarray(x, dtype=complex)
		N = x.shape[0]
		n = np.arange(N)
		k = n.reshape((N, 1))
		inter = -1*k*n
		M = np.exp(-2j * np.pi/N * inter)
		if np.array_equal(x,self.cv):
			return 1/N * np.dot(M, x)
		else:
			return np.dot(M,x)

	def inv_fft(self, x=None):
		if x is None:
			x = self.cv
		x = np.asarray(x, dtype=complex)
		N = x.shape[0]
		if N % 2 > 0:
			raise ValueError("must be a power of 2")
		elif N <= 2:
			return self.inv_dft(x)
		else:
			X_even = self.inv_fft(x[::2])
			X_odd = self.inv_fft(x[1::2])
			terms = np.exp(2j * np.pi/N * np.arange(N))
		if np.array_equal(x, self.cv):
			return 1/N * np.concatenate([X_even + terms[:int(N/2)]*X_odd, X_even + terms[int(N/2):]*X_odd])
		else:
			return np.concatenate([X_even + terms[:int(N/2)]*X_odd, X_even + terms[int(N/2):]*X_odd])
	
	def naive_convolution(self, B):
		A = self
		n, m = A.deg_bound, B.deg_bound
		p = n + m - 1
		C = polynomial(np.zeros(p))
		for i in range(n):
			for j in range(m):
				C.cv[i + j] += A.cv[i] * B.cv[j]
		return C
