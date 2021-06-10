from operator import xor
import sys, os,csv,math,argparse,glob,json
from turtle import width
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

exp_number = sys.argv[1]
folder     = sys.argv[2]


zeromx  = np.zeros((4,4))
onesmx  = np.zeros(( 4,4 ))
monesmx = np.zeros(( 4,4 ))


for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_number}/exp{exp_number}/*.json"):
    with open(file)  as infile:
        data       =  json.load(infile)
        for obj in data.values():

            datum = np.array(obj['contributions'])
            n = int(obj['n'])

            t1    = datum[0]
            t2    = datum[1]
            t3    = datum[2]
            t4    = datum[3]

            for row in range(0,4):
                for col in range(0,4):

                    if int(datum[row][col]) == 0.0:
                        zeromx[row][col] +=n
                    if int(datum[row][col]) == 1.0:
                        onesmx[row][col] +=n
                    if int(datum[row][col]) == -1.0:
                        monesmx[row][col] +=n

print(zeromx)
print(onesmx)
print(monesmx)
print(zeromx + onesmx + monesmx)


fig, ax = plt.subplots(4, 4, sharex='col', sharey='row')

for i in range(0,4):

    for j in range(0,4):

        ax[i,j].tick_params(
            axis        = 'x',   
            which       = 'both',
            bottom      = False, 
            top         = False, 
            labelbottom = False)

        # ax[i,j].tick_params(
        #     axis        = 'y',    
        #     which       = 'both', 
        #     bottom      = False,  
        #     top         = False,  
        #     labelbottom = False)

        frequency      = [monesmx[i,j],zeromx[i,j], onesmx[i,j]]
        contrib_values = ['-1','0','1']
        ax[i,j].bar(contrib_values,frequency, color=['red', 'blue', 'green'],width=0.3)


for t in range(4):

    ax[t,0].set_ylabel(f'Trait {t}', fontsize=18)
    ax[t,0].get_yaxis().set_ticks([])

for x in range(4):
    ax[0,x].set_title(f'Gene {x}', fontsize=18)

plt.suptitle(f"Experiment {exp_number}", fontsize=24)
plt.show()
# plt.savefig(f"Experiments {exp_number}.png", bbox_inches='tight')











