from operator import xor
import sys, os,csv,math,argparse
import numpy as np
from typing import  Callable, List
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob, os

exp_sym     = int( sys.argv[1] )
exp_asym    = int( sys.argv[2] )
fitall_sym  = 0
fitall_asym = 0

cs = 0
ca = 0


for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/staticpopv3/exp{exp_sym}/exp{exp_sym}/*.parquet"):
    data =  pd.read_parquet(file)
    # if np.array(data[f'{exp_sym}'])[len(np.array(data[f'{exp_sym}']))-1] == 1:
    fitall_sym     +=  np.array( data['fit'] )
    cs +=1

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/staticpopv3/exp{exp_asym}/exp{exp_asym}/*.parquet"):
    data       =  pd.read_parquet(file)

    fitall_asym     +=  np.array(data['fit'])
    ca +=1

fitall_sym                                =                       fitall_sym  /cs
fitall_asym                               =                       fitall_asym /ca
time                                      =                       np         . arange  (len(fitall_sym))

plt.plot(time , fitall_sym , label="Symmetric (Exp {})" . format     ( exp_sym  ) , c="orange" )
plt.plot(time , fitall_asym, label="Asymmetric (Exp {})" .format     ( exp_asym ) , c="blue" )

figure = plt.gcf()
plt.legend()
figure.set_size_inches(20,8)
plt.title("Average Fitness")
# plt.suptitle(f"Experiments {exp_sym},{exp_asym}",fontsize=20)
figure.text(0.5, 0.04, 'BD Process Iteration', ha='center', va='center')
# plt.show()


plt.savefig(f"Experiments {exp_sym},{exp_asym}.png", bbox_inches='tight')
