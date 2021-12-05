import numpy as np
from timeit import default_timer as timer
def next_pow_of_two(n):
	# incase n itself is power of 2
	n = n - 1
	while n & n - 1:
		n = n & n - 1 
	return n << 1

def time_exec(func):
	start = timer()
	func()
	end = timer()
	print('time taken: ['+str(end - start)+'s]\n')

def compare_with_numpy(x,y):
	if np.allclose(x,y):
		return 'PASSED'
	else:
		return 'FAILED'