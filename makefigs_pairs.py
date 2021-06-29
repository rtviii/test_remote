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

fitall_sym  = 0
fitall_asym = 0

cs = 0
ca = 0


for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_sym}/fitness_data/*.parquet"):

    data        = pd.read_parquet(file)
    fitall_sym += np.array(data['fit'])
    cs         += 1

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_asym}/fitness_data/*.parquet"):

    data         = pd.read_parquet(file)
    fitall_asym += np.array(data['fit'])
    ca          += 1

fitall_sym                                =                       fitall_sym  /cs
fitall_asym                               =                       fitall_asym /ca
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


plt.savefig(f"gpmut_20mil_{exp_sym}_v_{exp_asym}.png", bbox_inches='tight')

