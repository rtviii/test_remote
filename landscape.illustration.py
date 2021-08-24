from pprint import pprint
import random
from re import U
import timeit
from datetime import datetime
from functools import reduce 
import json
from time import time
import sys, os
from typing import Callable, List
from unicodedata import unidata_version
import matplotlib
import numpy as np
import math
import argparse
import pickle
import matplotlib.pyplot as plt


def getdata(fpath:str):
	with open(fpath, 'rb') as infile:
		d= np.array(pickle.load(infile))
		return  [d[:,1],d[:,2],d[:,3],d[:,4]]






cor_02   = getdata('./cor02/fitness_data/data0.pkl'   )
uncor_02 = getdata('./uncor0.2/fitness_data/data0.pkl')
cor1     = getdata('./cor1/fitness_data/data0.pkl'    )
uncor1   = getdata('./uncor1/fitness_data/data0.pkl'  )

# *---------------------------------------------------------------------------

x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

fig.suptitle('Movement of Fitness Optima Through Time', fontsize=20)


xlb = 'Time (1000 iterations)'
ylb = 'Position of Fitness Optimum'



m1 = np.array(cor1[0])
m2 = np.array(cor1[1])
m3 = np.array(cor1[2])
m4 = np.array(cor1[3])
time2 = np.arange(len(m1))

ax1.plot(time2,m1, label="Mean Factor 1", c="orange" , linewidth=1.5 , alpha =1   , linestyle='solid'                      )
ax1.plot(time2,m2, label="Mean Factor 2", c="black"  , linewidth=4   , alpha =1   , dash_capstyle='round', linestyle=":"      , dashes=(1,10 ) )
ax1.plot(time2,m3, label="Mean Factor 3", c="cyan"   , linewidth=3   , alpha =1   , linestyle='--'    ,  dashes=(6,20) )
ax1.plot(time2,m4, label="Mean Factor 4", c="darkseagreen"  , linewidth=10   , alpha =0.2 , linestyle='solid'                      )
ax1.set_xlabel(xlb,fontsize=14)
ax1.set_ylabel(ylb,fontsize=14)
ax1.set_title("Correlated | Increment = 1")
ax1.grid(alpha=0.5)


m1 = np.array(uncor1[0])
m2 = np.array(uncor1[1])
m3 = np.array(uncor1[2])
m4 = np.array(uncor1[3])
time2 = np.arange(len(m1))

ax2.plot(time2,m1, label="Mean Factor 1", c="orange" , linewidth=1.5 , alpha =1   , linestyle='solid'                      )
ax2.plot(time2,m2, label="Mean Factor 2", c="black"  , linewidth=1   , alpha =1   ,  linestyle="solid"   )
ax2.plot(time2,m3, label="Mean Factor 3", c="cyan"   , linewidth=1   , alpha =1   , linestyle="solid"    )
ax2.plot(time2,m4, label="Mean Factor 4", c="darkseagreen"  , linewidth=10   , alpha =0.2 , linestyle='solid'                      )
ax2.set_xlabel(xlb,fontsize=14)
ax2.set_ylabel(ylb,fontsize=14)
ax2.set_title("Uncorrelated | Increment = 1")
ax2.grid(alpha=0.5)

m1 = np.array(cor_02[0])
m2 = np.array(cor_02[1])
m3 = np.array(cor_02[2])
m4 = np.array(cor_02[3])
time2 = np.arange(len(m1))

ax3.plot(time2,m1, label="Mean Factor 1", c="orange" , linewidth=1.5 , alpha =1   , linestyle='solid'                      )
ax3.plot(time2,m2, label="Mean Factor 2", c="black"  , linewidth=4   , alpha =1   , dash_capstyle='round', linestyle=":"      , dashes=(1,10 ) )
ax3.plot(time2,m3, label="Mean Factor 3", c="cyan"   , linewidth=3   , alpha =1   , linestyle='--'    ,  dashes=(6,20) )
ax3.plot(time2,m4, label="Mean Factor 4", c="darkseagreen"  , linewidth=10   , alpha =0.2 , linestyle='solid'                      )
ax3.set_title("Correlated | Increment = 0.2")
ax3.set_xlabel(xlb,fontsize=14)
ax3.set_ylabel(ylb,fontsize=14)
ax3.grid(alpha=0.5)

m1 = np.array(uncor_02[0])
m2 = np.array(uncor_02[1])
m3 = np.array(uncor_02[2])
m4 = np.array(uncor_02[3])
time2 = np.arange(len(m1))
ax4.plot(time2,m1, label="Mean Factor 1", c="orange" , linewidth=1.5 , alpha =1   , linestyle='solid'                      )
ax4.plot(time2,m2, label="Mean Factor 2", c="black"  , linewidth=1   , alpha =1   ,  linestyle="solid"   )
ax4.plot(time2,m3, label="Mean Factor 3", c="cyan"   , linewidth=1   , alpha =1   , linestyle="solid"    )
ax4.plot(time2,m4, label="Mean Factor 4", c="darkseagreen"  , linewidth=10   , alpha =0.2 , linestyle='solid'                      )
ax4.set_title("Uncorrelated | Increment = 0.2")
ax4.set_xlabel(xlb,fontsize=14)
ax4.set_ylabel(ylb,fontsize=14)
ax4.grid(alpha=0.5)

for ax in fig.get_axes():
    ax.label_outer()



# m1 = np.array(universe.m1)
# m2 = np.array(universe.m2)
# m3 = np.array(universe.m3)
# m4 = np.array(universe.m4)
# time2 = np.arange(len(m1))

# plt.plot(time2,m1, label="Mean Factor 1", c="orange" , linewidth=1.5 , alpha =1   , linestyle='solid'                      )
# plt.plot(time2,m2, label="Mean Factor 2", c="black"  , linewidth=4   , alpha =1   , dash_capstyle='round', linestyle=":"      , dashes=(1,10 ) )
# # plt.plot(time2,m2, label="Mean Factor 2", c="black"  , linewidth=4   , alpha =1   , dash_capstyle='round', linestyle=":"      )
# # plt.plot(time2,m3, label="Mean Factor 3", c="cyan"   , linewidth=3   , alpha =1   , linestyle='--'    )
# plt.plot(time2,m3, label="Mean Factor 3", c="cyan"   , linewidth=3   , alpha =1   , linestyle='--'    ,  dashes=(6,20) )
# plt.plot(time2,m4, label="Mean Factor 4", c="darkseagreen"  , linewidth=10   , alpha =0.2 , linestyle='solid'                      )



plt.yticks(np.arange(-1,1,0.2))
# plt.legend()
plt.grid(alpha=0.5)

figure = plt.gcf()
ax = plt.gca()

# ax.set_facecolor('gainsboro')
# figure.set_facecolor('gainsboro')

# plt.title(f"Corellated Landscape Changes (Landscape Increment of {LS_INCREMENT})", fontsize=20)
figure.set_size_inches(20,8)
# figure.text(0.5, 0.04, f'Correlated changes in', ha='center', va='center')



plt.show()

