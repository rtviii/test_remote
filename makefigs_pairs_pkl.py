from operator import xor
import sys, os,csv,math,argparse
import numpy as np
from typing import  Callable, List
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob, os
import pickle

exp_sym  = int( sys.argv[1] )
exp_asym = int( sys.argv[2] )
folder   = str( sys.argv[3] )

fitall_sym  = 0
fitall_asym = 0

cs = 0
ca = 0

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_sym}/fitness_data/*.pkl"):
	data = []
	with open(file, 'rb') as  infile:
		data       =   pickle.load(infile)
	fitall_sym += np.array( data[:,0])
	cs         += 1

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_asym}/fitness_data/*.pkl"):
	data = []
	with open(file, 'rb') as  infile:
		data       =   pickle.load(infile)
		print(data)
	fitall_asym += np.array( data[:,0])
	ca          += 1

plt.title("Average Fitness")
plt.suptitle(f"Experiments {exp_sym},{exp_asym}",fontsize=20)
figure.text(0.5, 0.04, 'BD Process Iteration (every 1000 iterations)', ha='center', va='center')
plt.show()


# plt.savefig(f"PKL_10mil_{exp_sym}_v_{exp_asym}.png", bbox_inches='tight')

