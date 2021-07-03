import numpy as np


alleles  = np.array([2, 3, 2, 10])


coeffs = np.array([
	[1, 1, 1, 1],
	[2, 2, 2, 2],
	[4, 4, 4, 4],
	[5, 5, 5, 5],
])


np.sum(coeffs * ( alleles ** 1), axis=1)


print(coeffs * ( alleles ** 1))
print(np.sum(coeffs * ( alleles ** 1), axis=1))