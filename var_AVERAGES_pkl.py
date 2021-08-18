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
	return np.mean( np.array(data['covar_slices']) , axis=0)


def experiment(abs_root:str,number:int):

    cor   = []
    uncor = []


    varcovarfolder_sym  = os.path.join(abs_root, f"exp{number}",'var_covar')
    varcovarfolder_asym = os.path.join(abs_root, f"exp{number+1}",'var_covar')

    for file_sym in glob.glob(f"{varcovarfolder_sym}/*.pkl"):
        s    = get_variance_replicate(file_sym)
        scov = np.mean([
        s[0][1],
        s[0][2],
        s[0][3],
        s[1][2],
        s[1][3],
        s[2][3]
        ])
        svar = np.mean([
            s[0][0],
            s[1][1],
            s[2][2],
            s[3][3],
        ])
        cor.append([svar,scov])

    for file_asym in glob.glob(f"{varcovarfolder_asym}/*.pkl"):
        a    = get_variance_replicate(file_asym)
        scov = np.mean([
        a[0][1],
        a[0][2],
        a[0][3],
        a[1][2],
        a[1][3],
        a[2][3]
        ])
        svar = np.mean([
            a[0][0],
            a[1][1],
            a[2][2],
            a[3][3],
        ])
        uncor.append([svar,scov])

    sym  = np.array(cor)
    asym = np.array(uncor)

    # t        = np.around(   ttest_ind(sym ,asym  ), 10)
    # meansym  = np.around(np.mean     (sym ,axis=0), 10)
    # meanasym = np.around(np.mean     (asym,axis=0), 10)
    t        =    ttest_ind(sym ,asym  )
    meansym  = np.mean     (sym ,axis=0)
    meanasym = np.mean     (asym,axis=0)

    print(f"\t\t\033[91m P-values(Between {sym.shape[0]} replicates): \033[0m")
    pprint(t[1].tolist())
    
    print("\t\t\033[91m (Mean Variance, Mean Covariance) Corellated:\033[0m")
    pprint(meansym.tolist())

    print("\t\t\033[91m (Mean Variance, Mean Covariance) Uncorellated:\033[0m")
    pprint(meanasym.tolist())



abs_root =      sys.argv[1]
exp_sym  = int( sys.argv[2] )

#! exp_asym is exp_sym + 1
experiment(abs_root,exp_sym)

