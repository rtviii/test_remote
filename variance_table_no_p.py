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

abs_root = sys.argv[1]
exp_type = int( sys.argv[2] )

def get_variance_replicate(rep_path:str)-> List[np.ndarray]: 
	with open(rep_path, 'rb') as infile:
		data  = json.load(infile)

	mean_var   = np.array(data['var'])/data['elapsed']
	mean_covar = np.array(data['covar'])/data['elapsed']
	return [mean_var,mean_covar]

def experiment():

    # colnames         = ['type #','U_corellated' , 'U_uncorellated' , 'p_val']
    colnames         = ['type #','U_corellated' ]
    variance_sheet   = pd.DataFrame([], columns=colnames)
    covariance_sheet = pd.DataFrame([], columns=colnames)

    pvals_var   = []
    pvals_covar = []

    var_cor     = []
    cov_cor     = []

    var_uncor   = []
    cov_uncor   = []
    
    # for instance_file in os.listdir(os.path.join(abs_root, f"exp{exp_type}",'var_covar')):
    varcovarfolder =os.path.join(abs_root, f"exp{exp_type}",'var_covar')
    for file in glob.glob(f"{varcovarfolder}/*.json"):

        replicate_number = re.findall(r'\d+', file)[-1]
        instvar, instcovar = get_variance_replicate(file)

        var_cor.append(instvar)
        cov_cor.append(instcovar)

    #?----------- * VARIANCE * -----------

    """t1, t2, t3, t4"""
    mean_variance_correlated = np.mean(var_cor, axis=0)

    variance_sheet = variance_sheet.append(
    pd.DataFrame([[
                exp_type,
                str(np.around(mean_variance_correlated,5)),
                # str(np.around(mean_variance_uncorrelated,5)),
                # str(np.around([p_var_t1, p_var_t2, p_var_t3, p_var_t4],5))
            ]],columns=colnames)
            )


    mean_covariances_correlated  = np.mean(cov_cor,axis=0)


    covariance_sheet = covariance_sheet.append(pd.DataFrame([[exp_type, str(np.around(mean_covariances_correlated,5))]],columns=colnames))
    variance_sheet.to_csv('VAR_flat_large_increment.csv')
    covariance_sheet.to_csv('COV_flat_large_increment.csv')


experiment()
