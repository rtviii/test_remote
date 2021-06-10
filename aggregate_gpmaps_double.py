from operator import xor
import sys, os,csv,math,argparse,glob,json
from turtle import width
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

exp_sym  = sys.argv[1]
exp_asym = sys.argv[2]
folder   = sys.argv[3]

zeromx_sym  = np.zeros((4,4))
onesmx_sym  = np.zeros(( 4,4 ))
monesmx_sym = np.zeros(( 4,4 ))

zeromx_asym  = np.zeros((4,4))
onesmx_asym  = np.zeros(( 4,4 ))
monesmx_asym = np.zeros(( 4,4 ))

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_asym}/exp{exp_asym}/*.json"):
    with open(file)  as infile:
        data       =  json.load(infile)
        for obj in data.values():

            datum = np.array(obj['contributions'])
            n     = int(obj['n'])

            for row in range(0,4):
                for col in range(0,4):
                    if int(datum[row][col]) == 0.0:
                        zeromx_asym[row][col] += n
                    if int(datum[row][col]) == 1.0:
                        onesmx_asym[row][col] +=n
                    if int(datum[row][col]) == -1.0:
                        monesmx_asym[row][col] +=n

for file in glob.glob(f"/home/rxz/dev/polygenicity-simulations/{folder}/exp{exp_sym}/exp{exp_sym}/*.json"):
    with open(file)  as infile:
        data       =  json.load(infile)
        for obj in data.values():

            datum = np.array(obj['contributions'])
            n     = int(obj['n'])
            for row in range(0,4):
                for col in range(0,4):
                    if int(datum[row][col]) == 0.0:
                        zeromx_sym[row][col] +=n
                    if int(datum[row][col]) == 1.0:
                        onesmx_sym[row][col] +=n
                    if int(datum[row][col]) == -1.0:
                        monesmx_sym[row][col] +=n




fig, ax = plt.subplots(4, 4, sharex='col', sharey='row')

for i in range(0,4):
    for j in range(0,4):

        ax[i,j].tick_params(
            axis        = 'x',   
            which       = 'both',
            bottom      = False, 
            top         = False, 
            labelbottom = False)


        frequency_sym       = [monesmx_sym[i,j],zeromx_sym[i,j], onesmx_sym[i,j]]
        frequency_asym      = [monesmx_asym[i,j],zeromx_asym[i,j], onesmx_asym[i,j]]
        # contrib_values      = ['-1','0','1']
        contrib_values_sym  = [0, 1, 2]
        contrib_values_asym = [0.2, 1.2, 2.2]

        ax[i,j].bar(contrib_values_sym, frequency_sym,       color=['red', 'blue', 'green'],width=0.1, edgecolor="black")
        ax[i,j].bar(contrib_values_asym,frequency_asym,      color='white',                 width=0.1, edgecolor=['red', 'blue', 'green'])


for t in range(4):
    ax[t,0].set_ylabel(f'Trait {t}', fontsize=12)
    # ax[t,0].get_yaxis().set_ticks([])

for x in range(4):
    ax[0,x].set_title(f'Gene {x}', fontsize=12)

plt.suptitle(f"Experiments {exp_sym} & {exp_asym}", fontsize=14)
fig.set_size_inches(5, 5)
# plt.show()
plt.savefig(f"contrib_barplot_{exp_sym}_v_{exp_asym}.png", bbox_inches='tight', dpi=1200)











