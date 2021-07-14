import json
from operator import xor
import os
from pprint import pprint
from statistics import variance
import sys
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import pickle as pkl
from scipy.stats import ttest_ind


covs_cor   = np.array([])
covs_uncor = np.array([])

def get_variance_replicate(rep_path:str):

	with open(rep_path, 'rb') as infile:
		data  = pkl.load(infile)
		return np.cov(data['phenotype_agg'].T)
        

for file_s in glob.glob(f"/home/rxz/dev/polygenicity-simulations/tensimpl/exp1/var_covar/*.pkl"):
    v = get_variance_replicate(file_s)
    if covs_cor.shape == (0,):
        covs_cor = np.array( [ v ] )
    else:
        covs_cor = np.concatenate( ( covs_cor, np.array([ v ]) ))

for file_a in glob.glob(f"/home/rxz/dev/polygenicity-simulations/tensimpl/exp2/var_covar/*.pkl"):
    c = get_variance_replicate(file_a)
    if covs_uncor.shape == (0,):
        covs_uncor = np.array( [c ] )
    else:
        covs_uncor = np.concatenate( ( covs_uncor, np.array([ c]) ))

# print(covs_cor)
# print(covs_uncor)
print(covs_cor.shape)
print(covs_uncor.shape)

print("mean correlated:")
pprint(np.mean(covs_cor, axis=0))

print("mean uncorrelated:")
pprint(np.mean(covs_uncor, axis=0))
print("\t\tp-vals")
pprint(ttest_ind(covs_cor,covs_uncor, axis=0)[1])
# abs_root = sys.argv[1]
# pprint(get_variance_replicate('./tensimpl/exp1/var_covar/mean_var_covar_89.pkl'))

# variance_sheet  .to_csv('VAR_.csv', index=False)
# covariance_sheet.to_csv('COV_.csv', index=False)