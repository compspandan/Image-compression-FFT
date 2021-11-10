from polynomial import polynomial
import numpy as np


v1 = polynomial()
print(v1.cv)
d = v1.dft(v1.cv)
print(d)
print(np.allclose(d, np.fft.fft(v1.cv)))
d = v1.fft(v1.cv)
print(d)
print(np.allclose(d, np.fft.fft(v1.cv)))