from operator import xor
import sys, os,csv,math,argparse
import numpy as np
from typing import  Callable, List
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob, os

exp_sym  = int( sys.argv[1] )
exp_asym = int( sys.argv[2] )
folder   = str( sys.argv[3] )

fitall_sym  = []
fitall_asym = []

cs = 0
ca = 0


for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}_10mil/exp{exp_sym}/fitness_data/*.parquet"):

    instnance_n = re.findall(r'\d+', file)[-1]
    data_10mil        = pd.read_parquet(file)
    try:
        data_20mil = pd.read_parquet(f'/home/rxz/dev/polygenicity-simulations/{folder}_20mil/exp{exp_sym}/fitness_data/data{instnance_n}.parquet')
    except:
        print(f"Instance {instnance_n} is missing in 20 mil.")
        pass
    # fitall_sym += np.array(data_10mil['fit'])
    # fitall_sym += np.array(data_20mil['fit'])
    fitall_sym = np.append(data_10mil['fit'], data_20mil['fit'])
    cs         += 1

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}_10mil/exp{exp_asym}/fitness_data/*.parquet"):

    instnance_n = re.findall(r'\d+', file)[-1]
    data_10mil        = pd.read_parquet(file)
    try:
        data_20mil = pd.read_parquet(f'/home/rxz/dev/polygenicity-simulations/{folder}_20mil/exp{exp_asym}/fitness_data/data{instnance_n}.parquet')
    except:
        print(f"Instance {instnance_n} is missing in 20 mil.")
        pass

    fitall_asym = np.append(data_10mil['fit'], data_20mil['fit'])
    # fitall_asym += np.array(data_10mil['fit'])
    # fitall_asym += np.array(data_20mil['fit'])
    ca         += 1

# fitall_sym                                =                       fitall_sym  /cs
# fitall_asym                               =                       fitall_asym /ca
time                                      =                       np         . arange  (len(fitall_sym))

plt.plot(time , fitall_sym , label="Correlated (Exp {})" . format     ( exp_sym  ) , c="orange" )
plt.plot(time , fitall_asym, label="Uncorrelated (Exp {})".format     ( exp_asym ) , c="blue" )

figure = plt.gcf()
plt.legend()
figure.set_size_inches(20,8)
plt.title("Average Fitness")
plt.suptitle(f"Experiments {exp_sym},{exp_asym}",fontsize=20)
figure.text(0.5, 0.04, 'BD Process Iteration (1000 iterations)', ha='center', va='center')
# plt.show()


plt.savefig(f"flat_20mil_{exp_sym}_v_{exp_asym}.png", bbox_inches='tight')

