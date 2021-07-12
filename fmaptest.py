import math
import numpy as np
import pandas as pd


phenotypes = \
np.array([
	   [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.04, -0.98,  0.93,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02,  0.15,  1.03,  1.01],
       [-1.02, -0.8 ,  0.98,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-0.95, -1.59,  0.82,  0.71],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.04, -0.98,  0.93,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-0.99, -1.09,  0.99,  0.89],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-0.99, -1.09,  0.99,  0.89],
       [-0.99, -1.09,  0.99,  0.89],
       [-1.02, -0.93,  0.64,  1.  ],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-0.37, -0.98,  1.13,  1.03],
       [-1.05, -0.88,  1.01,  1.16],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.19,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-0.96, -0.98,  0.95,  1.01],
       [-0.99, -1.09,  0.99,  0.89],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.9 ,  0.53,  0.99],
       [-1.04, -0.99,  0.97,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-0.99, -1.09,  0.99,  0.89],
       [-0.19, -0.98,  1.23,  1.04],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-0.99, -1.09,  0.99,  0.89],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.04, -0.92,  1.02,  1.1 ],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.17, -0.52,  0.92,  1.59],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-0.99, -1.05,  0.81,  0.88],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.05, -0.87,  1.01,  1.17],
       [-1.02, -1.  ,  1.03,  1.01],
       [-0.97, -1.16,  1.01,  0.8 ],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.39, -1.  ,  0.91,  0.99],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.05, -0.88,  1.01,  1.16],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.8 ,  0.98,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-0.99, -1.09,  0.99,  0.89],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.05, -0.88,  1.01,  1.16],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.5 ,  0.98,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-0.33, -0.99,  1.24,  1.03],
       [-1.04, -0.99,  0.97,  1.01],
       [-1.02, -0.99,  0.98,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.05,  1.28,  1.02],
       [-1.04, -0.98,  0.93,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.04, -0.98,  0.93,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01],
       [-1.02, -1.  ,  1.03,  1.01]])

print(np.cov(phenotypes, rowvar=False))


data = pd.DataFrame({
	'trait1': phenotypes[:, 0],
 	'trait2': phenotypes[:, 1], 
 	'trait3': phenotypes[:, 2], 
 	'trait4': phenotypes[:, 3], 
	})

print(data)
print("correltaion", data.corr(method='pearson'))