from statistics import mode
import sys, os,csv,math,argparse,glob
import numpy as np
import argparse
import pandas as pd
from scipy.stats import sem




df = pd.DataFrame(      columns    =[    'experiment_number'             , 'standard_error' ])
for exp in range( 1,42 ):

    fitness_values = 0
    c              = 0
    for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/staticpopv3/exp{exp}/exp{exp}/*.parquet"):
        data            = pd.read_parquet(file)
        fitness_values += np.array( data['fit'] )
        c              += 1

    fitness_values /= c
    std_erorr = sem(fitness_values)

    df.loc[exp] = [f'experiment_{exp}', std_erorr]

df.to_csv   ('std_error.csv',mode='a')