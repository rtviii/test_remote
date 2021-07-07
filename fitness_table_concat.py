from operator import xor
from pprint import pprint
import sys, os,csv,math,argparse
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
from scipy.stats import ttest_ind

# exp_sym  = sys.argv[1]
# TYPE     = exp_sym
folder   = "staticpop_lite"
# exp_asym = sys.argv[2]



#? for each experiment:
#? 100 x the mean of the last third correlated    -|__>>> pvalue, mean of 100means cor, mean of 100 means uncorr
#? 100 x the mean of the last third uncorrelated  -|


correlated_exps   = []
uncorrelated_exps = []
pvalues           = []

for exp_type in range(11,14):

    fitness_sym  = []
    fitness_asym = []

    print("on Experiment ", exp_type)


    for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_type}/fitness_data/*.parquet"):
        data      = pd.read_parquet(file)
        lastthird = data['fit'][   int( len(data['fit']) - len(data['fit'])/3 ):]
        fit_sym_u = np.mean(lastthird)
        fitness_sym.append(fit_sym_u)

    for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_type+5}/fitness_data/*.parquet"):
        data       = pd.read_parquet(file)
        lastthird  = data['fit'][   int( len(data['fit']) - len(data['fit'])/3 ):]
        fit_asym_u = np.mean(lastthird)
        fitness_asym.append(fit_asym_u)


    pvalue            = ttest_ind(fitness_sym, fitness_asym     )[1]
    pvalues          .  append   (pvalue                        )
    correlated_exps  .  append   (np         .mean(fitness_sym) )
    uncorrelated_exps.  append   (np         .mean(fitness_asym))


print(correlated_exps)
print(uncorrelated_exps)
# print(cwpuncorrelated_exps)

df = pd.DataFrame({'correlated':correlated_exps,'uncorrelated':uncorrelated_exps,'p-val':pvalues})
print(df)
df.to_csv("FITNESS_small_increment.csv")



# if bool(casym - csym):
#     print("Unequal numbers of datapoints :", casym, csym)
# table = pd.DataFrame(data= np.array([fitness_sym, fitness_asym]).transpose(),index   = np.arange(len(fitness_asym)),
# columns = [f"exp_{exp_sym}_SYM",f"exp_{exp_asym}_ASYM"])
# print(table)
# filename =f"final_fitness_gpmut_{exp_sym}_{exp_asym}.csv"
# table.to_csv(filename)
# print("Generated file at {}".format(filename))