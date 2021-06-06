import sys, os,csv,math,argparse
import numpy as np
import argparse
import pandas as pd
import scipy
from scipy.stats import ttest_ind



experiments = []
tvals       = []


for i in range(1,18):
    if i in [ 3,12,15 ]:
        continue
    df           = pd.read_csv('final_fitness_{}_{}.csv'.format(i, i+18))
    correlated   = df[f'exp_{i}_SYM']
    uncorrelated = df[f'exp_{i+18}_ASYM']
    x =ttest_ind(correlated,uncorrelated)
    experiments.append("{},{}".format(i,i+18))
    tvals.append(x[1])

    


for i in [37,38,39]:
    df           = pd.read_csv('final_fitness_{}_{}.csv'.format(i, i+3))
    correlated   = df[f'exp_{i}_SYM']
    uncorrelated = df[f'exp_{i+3}_ASYM']
    x = ttest_ind(correlated,uncorrelated)
    experiments.append("{},{}".format(i,i+3))
    tvals.append(x[1])

pd.DataFrame(data=np.array([ experiments,tvals ]).transpose(), columns=["exp-pair","tval"], index=np.arange(len(experiments))).to_csv("tvals.csv")