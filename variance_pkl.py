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
from scipy.stats import ttest_ind



def get_variance_replicate(rep_path:str)-> List[np.ndarray]: 

	with open(rep_path, 'rb') as infile:
		data  = json.load(infile)

	mean_var   = np.array(data['var'])/data['elapsed']
	mean_covar = np.array(data['covar'])/data['elapsed']
	return [mean_var,mean_covar]


def experiment(number, variance_sheet, covariance_sheet):


    pvals_var   = []
    pvals_covar = []

    var_cor     = []
    cov_cor     = []

    var_uncor   = []
    cov_uncor   = []

    varcovarfolder_sym  = os.path.join(abs_root, f"exp{number}",'var_covar')
    varcovarfolder_asym = os.path.join(abs_root, f"exp{number+1}",'var_covar')

    for file in glob.glob(f"{varcovarfolder_sym}/*.json"):

        replicate_number = re.findall(r'\d+', file)[-1]
        instvar, instcovar = get_variance_replicate(file)

        var_cor.append(instvar)
        cov_cor.append(instcovar)

    for file in glob.glob(f"{varcovarfolder_asym}/*.json"):

        replicate_number = re.findall(r'\d+', file)[-1]
        instvar, instcovar = get_variance_replicate(file)

        var_uncor.append(instvar)
        cov_uncor.append(instcovar)



    """t1, t2, t3, t4"""
    variance_sym  = np.array(var_cor)
    variance_asym = np.array(var_uncor)


    smaller = None
    if len(variance_sym)    > len(variance_asym):
        smaller = len(variance_asym)
    else:
        smaller = len(variance_sym) 




    variance_sym= variance_sym[:smaller]
    variance_asym= variance_asym[:smaller]


    # print(f"variance sample for exp {exp_type}"  , len(variance_sym ))
    # print(f"variance sample for exp {exp_type+1}", len(variance_asym))

    # print("Smaller:", smaller)
    # print("trucnated exps:", len(variance_sym[:smaller]))
    # print("trucnated expa:", len(variance_asym[:smaller]))


    p_var_t1 = ttest_ind(variance_sym[:,0], variance_asym[:,0])[1]
    p_var_t2 = ttest_ind(variance_sym[:,1], variance_asym[:,1])[1]
    p_var_t3 = ttest_ind(variance_sym[:,2], variance_asym[:,2])[1]
    p_var_t4 = ttest_ind(variance_sym[:,3], variance_asym[:,3])[1]

    mean_variance_correlated   = np.mean(variance_sym, axis=0)
    mean_variance_uncorrelated = np.mean(variance_asym, axis=0)

    variance_sheet = variance_sheet.append(

    pd.DataFrame([[
                number,
                str(np.around(mean_variance_correlated,5)),
                str(np.around(mean_variance_uncorrelated,5)),
                str(np.around([p_var_t1, p_var_t2, p_var_t3, p_var_t4],5))
            ]],columns=colnames)
            )

    """t1_t2, t1_t3, t1_t4, t2_t3, t2_t4, t3_t4"""
    covariance_sym  = np.array(cov_cor)[:smaller]
    covariance_asym = np.array(cov_uncor)[:smaller]

    p_t1_t2         = ttest_ind(covariance_sym[0], covariance_asym[0])[1]
    p_t1_t3         = ttest_ind(covariance_sym[1], covariance_asym[1])[1]
    p_t1_t4         = ttest_ind(covariance_sym[2], covariance_asym[2])[1]
    p_t2_t3         = ttest_ind(covariance_sym[3], covariance_asym[3])[1]
    p_t2_t4         = ttest_ind(covariance_sym[4], covariance_asym[4])[1]
    p_t3_t4         = ttest_ind(covariance_sym[5], covariance_asym[5])[1]

    pvals_covar     = [
        p_t1_t2,
        p_t1_t3,
        p_t1_t4,
        p_t2_t3,
        p_t2_t4,
        p_t3_t4
        ]

    covariance_sheet = covariance_sheet.append(pd.DataFrame([
         [number, str(np.around(mean_variance_correlated,5)), str(np.around(mean_variance_uncorrelated,5)) ,str(np.around(pvals_covar,5))] 
         ],columns=colnames, ))



abs_root = sys.argv[1]
# exp_type = int( sys.argv[2] )
colnames         = ['type #','U_corellated' , 'U_uncorellated' , 'p_val']
variance_sheet   = pd.DataFrame([], columns=colnames)
covariance_sheet = pd.DataFrame([], columns=colnames)

for number in [1,3,5,7,9,11,13,15]:
    pvals_var   = []
    pvals_covar = []

    var_cor     = []
    cov_cor     = []

    var_uncor   = []
    cov_uncor   = []

    varcovarfolder_sym  = os.path.join(abs_root, f"exp{number}",'var_covar')
    varcovarfolder_asym = os.path.join(abs_root, f"exp{number+1}",'var_covar')

    for file in glob.glob(f"{varcovarfolder_sym}/*.json"):

        replicate_number = re.findall(r'\d+', file)[-1]
        instvar, instcovar = get_variance_replicate(file)

        var_cor.append(instvar)
        cov_cor.append(instcovar)

    for file in glob.glob(f"{varcovarfolder_asym}/*.json"):

        replicate_number = re.findall(r'\d+', file)[-1]
        instvar, instcovar = get_variance_replicate(file)

        var_uncor.append(instvar)
        cov_uncor.append(instcovar)



    """t1, t2, t3, t4"""
    variance_sym  = np.array(var_cor)
    variance_asym = np.array(var_uncor)


    smaller = None
    if len(variance_sym)    > len(variance_asym):
        smaller = len(variance_asym)
    else:
        smaller = len(variance_sym) 




    variance_sym= variance_sym[:smaller]
    variance_asym= variance_asym[:smaller]


    # print(f"variance sample for exp {exp_type}"  , len(variance_sym ))
    # print(f"variance sample for exp {exp_type+1}", len(variance_asym))

    # print("Smaller:", smaller)
    # print("trucnated exps:", len(variance_sym[:smaller]))
    # print("trucnated expa:", len(variance_asym[:smaller]))


    p_var_t1 = ttest_ind(variance_sym[:,0], variance_asym[:,0])[1]
    p_var_t2 = ttest_ind(variance_sym[:,1], variance_asym[:,1])[1]
    p_var_t3 = ttest_ind(variance_sym[:,2], variance_asym[:,2])[1]
    p_var_t4 = ttest_ind(variance_sym[:,3], variance_asym[:,3])[1]

    mean_variance_correlated   = np.mean(variance_sym, axis=0)
    mean_variance_uncorrelated = np.mean(variance_asym, axis=0)

    variance_sheet = variance_sheet.append(

    pd.DataFrame([[
                number,
                str(np.around(mean_variance_correlated,5)),
                str(np.around(mean_variance_uncorrelated,5)),
                str(np.around([p_var_t1, p_var_t2, p_var_t3, p_var_t4],5))
            ]],columns=colnames)
            )

    """t1_t2, t1_t3, t1_t4, t2_t3, t2_t4, t3_t4"""
    covariance_sym  = np.array(cov_cor)[:smaller]
    covariance_asym = np.array(cov_uncor)[:smaller]

    mean_covariance_c = np.mean(covariance_sym, axis=0)
    mean_covariance_u = np.mean(covariance_asym, axis=0)
    p_t1_t2         = ttest_ind(covariance_sym[0], covariance_asym[0])[1]
    p_t1_t3         = ttest_ind(covariance_sym[1], covariance_asym[1])[1]
    p_t1_t4         = ttest_ind(covariance_sym[2], covariance_asym[2])[1]
    p_t2_t3         = ttest_ind(covariance_sym[3], covariance_asym[3])[1]
    p_t2_t4         = ttest_ind(covariance_sym[4], covariance_asym[4])[1]
    p_t3_t4         = ttest_ind(covariance_sym[5], covariance_asym[5])[1]
    pvals_covar     = [
        p_t1_t2,
        p_t1_t3,
        p_t1_t4,
        p_t2_t3,
        p_t2_t4,
        p_t3_t4
        ]



    covariance_sheet = covariance_sheet.append(pd.DataFrame([
         [number, str(np.around(mean_covariance_c,5)), str(np.around(mean_covariance_u,5)) ,str(np.around(pvals_covar,5))] 
         ],columns=colnames, ))

variance_sheet  .to_csv('VAR_.csv', index=False)
covariance_sheet.to_csv('COV_.csv', index=False)
