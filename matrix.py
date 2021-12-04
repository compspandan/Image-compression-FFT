import numpy as np

from polynomial import polynomial
class matrix:
	max_val = 100
	def __init__(self,m=None):
		if m is None:
			self.n = 4
			self.matrix = np.random.randint(1, self.max_val, (self.n,self.n))
		else:
			self.matrix = m
	
	def fft_2D(self):
		fft_matrix = np.zeros(self.matrix.shape,dtype=complex)
		for i,row in enumerate(self.matrix):
			poly = polynomial(row)
			fft = poly.fft()
			fft_matrix[i] = fft
		for i,col in enumerate(fft_matrix.T):
			poly = polynomial(col)
			fft = poly.fft()
			fft_matrix[:,i] = fft
		return fft_matrix
	
	def ifft_2D(self):
		ifft_matrix = np.zeros(self.matrix.shape,dtype=complex)
		for i,row in enumerate(self.matrix):
			poly = polynomial(row)
			ifft = poly.inv_fft()
			ifft_matrix[i] = ifft
		for i,col in enumerate(ifft_matrix.T):
			poly = polynomial(col)
			ifft = poly.inv_fft()
			ifft_matrix[:,i] = ifft
		return ifft_matrix