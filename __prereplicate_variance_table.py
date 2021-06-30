import json
from operator import xor
from pprint import pprint
from statistics import variance
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from scipy.stats import ttest_ind

  

folder="flat_pop8_20mil"


colnames         = ['type #','U_corellated' , 'U_uncorellated' , 'p_val']
variance_sheet   = pd.DataFrame([], columns=colnames)
covariance_sheet = pd.DataFrame([], columns=colnames)

for exp_type in range(1,6):

    pvals_var   = []
    pvals_covar = []

    var_cor   = []
    cov_cor   = []

    var_uncor = []
    cov_uncor = []

    for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_type}/var_covar/*.json"):
        dump_itern = re.findall(r'\d+', file)[-1]
        if int(dump_itern) > int(17e6):
            with open (file ) as infile:
                data = json.load(infile)
                var  = data['variance']
                cov  = data['covariance']

                var_cor.append(var)
                cov_cor.append(cov)

    for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_type+5}/var_covar/*.json"):
        dump_itern = re.findall(r'\d+', file)[-1]
        if int(dump_itern) > int(17e6):
            with open (file ) as infile:
                data = json.load(infile)
                var  = data['variance']
                cov  = data['covariance']

                var_uncor.append(var)
                cov_uncor.append(cov)

    #?----------- * VARIANCE * -----------

    """t1, t2, t3, t4"""
    variance_sym  = np.array(var_cor)
    variance_asym = np.array(var_uncor)

    print(variance_sym)
    p_var_t1 = ttest_ind(variance_sym[:,0], variance_asym[:,0])[1]
    p_var_t2 = ttest_ind(variance_sym[:,1], variance_asym[:,1])[1]
    p_var_t3 = ttest_ind(variance_sym[:,2], variance_asym[:,2])[1]
    p_var_t4 = ttest_ind(variance_sym[:,3], variance_asym[:,3])[1]

    mean_variance_correlated   = np.mean(variance_sym, axis=0)
    mean_variance_uncorrelated = np.mean(variance_asym, axis=0)

    variance_sheet =variance_sheet.append(
    pd.DataFrame(
            [ [
                exp_type,
                str(np.around(mean_variance_correlated,5)),
                str(np.around(mean_variance_uncorrelated,5)),
                str(np.around([p_var_t1, p_var_t2, p_var_t3, p_var_t4],5))
            ] ]
        ,columns=colnames)
    )

    #?----------- * COVARIANCE * -----------

    """t1_t2, t1_t3, t1_t4, t2_t3, t2_t4, t3_t4"""
    covariance_sym  = np.array(cov_cor)
    covariance_asym = np.array(cov_uncor)

    p_t1_t2         = ttest_ind(covariance_sym[:,0,1], covariance_asym[:,0,1])[1]
    p_t1_t3         = ttest_ind(covariance_sym[:,0,2], covariance_asym[:,0,2])[1]
    p_t1_t4         = ttest_ind(covariance_sym[:,0,3], covariance_asym[:,0,3])[1]
    p_t2_t3         = ttest_ind(covariance_sym[:,1,2], covariance_asym[:,1,2])[1]
    p_t2_t4         = ttest_ind(covariance_sym[:,1,3], covariance_asym[:,1,3])[1]
    p_t3_t4         = ttest_ind(covariance_sym[:,2,3], covariance_asym[:,2,3])[1]

    pvals_covar     = [

        p_t1_t2,
        p_t1_t3,
        p_t1_t4,
        p_t2_t3,
        p_t2_t4,
        p_t3_t4

        ]

    mean_covariances_correlated  = np.mean(covariance_sym,axis=0)
    mean_covariances_uncorrelated =np.mean(covariance_asym,axis=0)

    #! mcct_t_ for MEAN COVARIANCE CORELLATED TRAIT X TRAIT Y
    mcct1t2 = mean_covariances_correlated[0,1]
    mcct1t3 = mean_covariances_correlated[0,2]
    mcct1t4 = mean_covariances_correlated[0,3]
    mcct2t3 = mean_covariances_correlated[1,2]
    mcct2t4 = mean_covariances_correlated[1,3]
    mcct3t4 = mean_covariances_correlated[2,3]
    mean_covs_C =[ 
    mcct1t2,
    mcct1t3,
    mcct1t4,
    mcct2t3,
    mcct2t4,
    mcct3t4
    ]
    

    #! mcct_t_ for MEAN COVARIANCE UNCORELLATED TRAIT X TRAIT Y
    mcut1t2 = mean_covariances_uncorrelated[0,1]
    mcut1t3 = mean_covariances_uncorrelated[0,2]
    mcut1t4 = mean_covariances_uncorrelated[0,3]
    mcut2t3 = mean_covariances_uncorrelated[1,2]
    mcut2t4 = mean_covariances_uncorrelated[1,3]
    mcut3t4 = mean_covariances_uncorrelated[2,3]
    mean_covs_U =[ 
    mcut1t2,
    mcut1t3,
    mcut1t4,
    mcut2t3,
    mcut2t4,
    mcut3t4
    ]

    covariance_sheet =covariance_sheet.append(pd.DataFrame([ [exp_type, str(np.around(mean_covs_C,5)), str(np.around(mean_covs_U,5)) ,str(np.around(pvals_covar,5))] ],columns=colnames))


variance_sheet.to_csv('VAR_flat_large_increment.csv')
covariance_sheet.to_csv('COV_flat_large_increment.csv')
