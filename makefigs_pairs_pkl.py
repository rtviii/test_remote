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
title    = str( sys.argv[4] )

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

fitall_sym                                =                       fitall_sym  /cs
fitall_asym                               =                       fitall_asym /ca
time                                      =                       np         . arange  (len(fitall_sym))

# plt.plot(time , fitall_sym , label="Correlated (Exp {})" . format     ( exp_sym  ) , c="black" )
# plt.plot(time , fitall_asym, label="Uncorrelated (Exp {})".format     ( exp_asym ) , c="orange",linestyle='dotted' )

plt.plot(time , fitall_sym , label="Correlated Landscape Shifts " . format     ( exp_sym  ) , c="black" )
plt.plot(time , fitall_asym, label="Uncorrelated Landscape Shifts".format     ( exp_asym ) , c="orange",linestyle='dotted' )

figure = plt.gcf()
plt.legend(loc='lower right')
figure.set_size_inches(20,8)
plt.grid(alpha=0.5)
figure = plt.gcf()
plt.ylabel('Average Fitness', fontsize=14)
plt.xlabel('Generations (generation = 1000 iterations)', fontsize=14)
# plt.title("Average Fitness")
# plt.suptitle(title,fontsize=20)
# figure.text(0.5, 0.04, 'Generations (generation = 1000 iterations)', ha='center', va='center')
plt.show()


# plt.savefig(f"PKL_10mil_{exp_sym}_v_{exp_asym}.png", bbox_inches='tight')

