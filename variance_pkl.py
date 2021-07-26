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

    asym  = np.array(cov_uncor)
    sym = np.array(cov_cor)



    t = ttest_ind(sym,asym)

    print("P-values")
    print(t[1])

    meansym  = np.mean(sym,axis=0)
    meanasym = np.mean(asym,axis=0)
    
    print("Corellated:")
    pprint(meansym)

    print("Uncorellated:")
    pprint(meanasym)



    # smaller = None
    # if len(variance_sym)    > len(variance_asym):
    #     smaller = len(variance_asym)
    # else:
    #     smaller = len(variance_sym) 

    # variance_sym= variance_sym[:smaller]
    # variance_asym= variance_asym[:smaller]


    #!----------------



    # variance_sheet = variance_sheet.append(

    # pd.DataFrame([[
    #             number,
    #             str(np.around(mean_variance_correlated,5)),
    #             str(np.around(mean_variance_uncorrelated,5)),
    #             str(np.around([p_var_t1, p_var_t2, p_var_t3, p_var_t4],5))
    #         ]],columns=colnames)
    #         )

    # """t1_t2, t1_t3, t1_t4, t2_t3, t2_t4, t3_t4"""
    # covariance_sym  = np.array(cov_cor)[:smaller]
    # covariance_asym = np.array(cov_uncor)[:smaller]

    # p_t1_t2         = ttest_ind(covariance_sym[0], covariance_asym[0])[1]
    # p_t1_t3         = ttest_ind(covariance_sym[1], covariance_asym[1])[1]
    # p_t1_t4         = ttest_ind(covariance_sym[2], covariance_asym[2])[1]
    # p_t2_t3         = ttest_ind(covariance_sym[3], covariance_asym[3])[1]
    # p_t2_t4         = ttest_ind(covariance_sym[4], covariance_asym[4])[1]
    # p_t3_t4         = ttest_ind(covariance_sym[5], covariance_asym[5])[1]

    # pvals_covar     = [
    #     p_t1_t2,
    #     p_t1_t3,
    #     p_t1_t4,
    #     p_t2_t3,
    #     p_t2_t4,
    #     p_t3_t4
    #     ]

    # covariance_sheet = covariance_sheet.append(pd.DataFrame([
    #      [number, str(np.around(mean_variance_correlated,5)), str(np.around(mean_variance_uncorrelated,5)) ,str(np.around(pvals_covar,5))] 
    #      ],columns=colnames, ))

abs_root = sys.argv[1]

experiment(abs_root,1)



# variance_sheet  .to_csv('VAR_.csv', index=False)
# covariance_sheet.to_csv('COV_.csv', index=False)
