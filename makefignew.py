import pickle 
from operator import xor
import sys, os,csv,math,argparse,glob
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt



exp_number = sys.argv[1]
folder     = sys.argv[2]
save_show = str(sys.argv[3])

mean1all  =  0
mean2all  =  0
mean3all  =  0
mean4all  =  0

fitall    =  0
brateall  =  0

c = 0


for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/fitness_data/*.csv"):

	print("Got pickle file ", file)

	data = []
	with open(file, 'rb') as  infile:
		data       =   pickle.load(infile)

	fitall     =  np.array( data[:,0])
	mean1all   =  np.array( data[:,1])
	mean2all   =  np.array( data[:,2])
	mean3all   =  np.array( data[:,3])
	mean4all   =  np.array( data[:,4])
	c +=1

mean1all  =  mean1all/c
mean2all  =  mean2all/c
mean3all  =  mean3all/c
mean4all  =  mean4all/c
fitall    =  fitall  /c


time          =  np.arange(len(fitall))
time2 = np.arange(len(mean1all))
figur, axarr  =  plt.subplots(2,2)

axarr[0,1].plot(time, fitall, label="Fitness")
axarr[0,1].set_ylabel('Population wide Fitness')

time2 = np.arange(len(mean1all))

axarr[1,0].plot(time2,mean1all, label="Mean 1", c="cyan")
axarr[1,0].plot(time2,mean2all, label="Mean 2", c="black")
axarr[1,0].plot(time2,mean3all, label="Mean 3", c="brown")
axarr[1,0].plot(time2,mean4all, label="Mean 4", c="yellow")
axarr[1,0].legend()

figure = plt.gcf()
figure.set_size_inches(20,8)
plt.suptitle(f"Experiment {exp_number}")
figure.text(0.5, 0.04, f'BD Process Iteration(every {1000} iterations)', ha='center', va='center')



if save_show == "+":
    plt.show()
if save_show == "_":
    plt.savefig(f"{folder}_exp{exp_number}.png")
    print(f"saved {folder}_exp{exp_number}.png")

