from operator import xor
import sys, os,csv,math,argparse,glob
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

exp_number = sys.argv[1]
folder     = sys.argv[2]

mean0all  =  0
mean1all  =  0
mean2all  =  0
mean3all  =  0

fitall    =  0
brateall  =  0
countall  =  0

c = 0

# for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_number}/fitness_data/*.parquet"):

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/fitness_data/*.parquet"):

    data       =   pd.read_parquet(file)
    mean0all   +=  np.array( data['mean0'])
    mean1all   +=  np.array( data['mean1'])
    mean2all   +=  np.array( data['mean2'])
    mean3all   +=  np.array( data['mean3'])
    fitall     +=  np.array( data['fit'])
    c+=1

mean0all  =  mean0all/c
mean1all  =  mean1all/c
mean2all  =  mean2all/c
mean3all  =  mean3all/c
fitall    =  fitall  /c
brateall  =  brateall/c
countall  =  countall/c

time          =  np.arange(len(fitall))
figur, axarr  =  plt.subplots(2,2)

axarr[0,1].plot(time, fitall, label="Fitness")
axarr[0,1].set_ylabel('Population wide Fitness')

time2 = np.arange(len(mean0all))

axarr[1,0].plot(time2,mean0all, label="Mean 1", c="cyan")
axarr[1,0].plot(time2,mean1all, label="Mean 2", c="black")
axarr[1,0].plot(time2,mean2all, label="Mean 3", c="brown")
axarr[1,0].plot(time2,mean3all, label="Mean 4", c="yellow")
axarr[1,0].legend()

figure = plt.gcf()
figure.set_size_inches(20,8)
plt.suptitle(f"Experiment {exp_number}")
figure.text(0.5, 0.04, f'BD Process Iteration(every {1000} iterations)', ha='center', va='center')

# plt.show()

plt.savefig("Experiment {exp_number} ")
