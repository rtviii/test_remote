from pprint import pprint
import numpy as np
mask = np.random.choice([0,1], size=(20,)).reshape((5,4))
print("mask happens to be:",mask)

alleles1 = np.array([
	1,1,1,1
])
alleles2 = np.array([
	5,5,5,5
])

contribs1=np.array([
	[1,1,1,1],
	[1,1,1,1],
	[1,1,1,1],
	[1,1,1,1]
])

contribs2=np.array([
	[8,8,8,8],
	[8,8,8,8],
	[8,8,8,8],
	[8,8,8,8]
]) +1

alleles1[mask[:1][0]==1] = alleles2[mask[:1][0]==1]

pprint(mask)

child =np.copy(contribs1)
print(child)
child[mask[1:]==1] = contribs2[mask[1:]==1]
print(child)
print(alleles1)
# child[]