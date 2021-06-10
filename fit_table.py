from operator import xor
import sys, os,csv,math,argparse
import numpy as np
from typing import  Callable, List
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob, os

exp_sym  = sys.argv[1]
exp_asym = sys.argv[2]
folder   = sys.argv[3]

fitness_sym  = []
fitness_asym = []


csym  = 0
casym = 0

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_sym}/exp{exp_sym}/*.parquet"):

    data =  pd.read_parquet(file)
    terminal     = data['fit'][len(data['fit'])-1]
    fitness_sym.append(  terminal)
    csym        += 1

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_asym}/exp{exp_asym}/*.parquet"):
    data       =  pd.read_parquet(file)
    terminal = data['fit'][len(data['fit'])-1]
    fitness_asym.append(  terminal)
    casym +=1

if bool(casym - csym):
    print("Unequal numbers of datapoints :", casym, csym)
table = pd.DataFrame(data= np.array([fitness_sym, fitness_asym]).transpose(),index   = np.arange(len(fitness_asym)),
columns = [f"exp_{exp_sym}_SYM",f"exp_{exp_asym}_ASYM"])
print(table)
filename =f"final_fitness_gpmut_{exp_sym}_{exp_asym}.csv"
table.to_csv(filename)
print("Generated file at {}".format(filename))


