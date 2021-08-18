import json
from operator import xor
import os
from pprint import pprint
from statistics import mean, variance
import sys
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import pickle as pkl
from scipy.stats import ttest_ind



def get_variance_replicate(rep_path:str):
	with open(rep_path, 'rb') as infile:
		data  = pkl.load(infile)

    #return the avergae covariance for the 
	return np.mean( np.array(data['covar_slices']) , axis=0)


def experiment(abs_root:str,number:int):

    cov_cor   = []
    cov_uncor = []

    varcovarfolder_sym  = os.path.join(abs_root, f"exp{number}",'var_covar')
    varcovarfolder_asym = os.path.join(abs_root, f"exp{number+1}",'var_covar')

    for file_sym in glob.glob(f"{varcovarfolder_sym}/*.pkl"):
        symcov           = get_variance_replicate(file_sym)
        cov_cor.append(symcov)

    for file_asym in glob.glob(f"{varcovarfolder_asym}/*.pkl"):
        asymcov          = get_variance_replicate(file_asym)
        cov_uncor.append(asymcov)

    asym = np.array(cov_uncor)
    sym  = np.array(cov_cor)

    t = ttest_ind(sym,asym)

    print("P-values")
    print(t[1])

    meansym  = np.mean(sym,axis=0)
    meanasym = np.mean(asym,axis=0)
    
    print("Corellated:")
    pprint(meansym)

    print("Uncorellated:")
    pprint(meanasym)



abs_root = sys.argv[1]
experiment(abs_root,7)

