from __future__ import annotations
from functools import reduce 
import json
from pprint import pprint
import xxhash
import sys, os
import numpy as np
from typing import  Callable, Dict, List, Tuple
import typing
import math
import argparse
import pandas as pd

VERBOSE = False

def dir_path(string):
    if string == "0":
        return None
    if os.path.isdir(string):
        return string
    else:
        try:
            if not os.path.exists(string):
                os.makedirs(string, exist_ok=True)
                return string
        except:
            raise PermissionError(string)

parser = argparse.ArgumentParser(description='Simulation presets')
parser.add_argument('-save',    '--outdir',              type=dir_path, help="""Specify the path to write the results of the simulation.""")
parser.add_argument("-it",      "--itern",               type=int,      help="The number of iterations")
parser.add_argument("-sim",     "--siminst",             type=int,      help="Simulation tag for the current instance.")
parser.add_argument("-SP",      "--shifting_peak",       type=int,      choices=[-1,0,1], help="Flag for whether the fitness landscape changes or not.")
parser.add_argument("-plot",    "--toplot",              type=int,      choices=[0,1])
parser.add_argument("-con",     "--connectivity",        type=int,      choices=[0,1])
parser.add_argument("-exp",     "--experiment",          type=int)
parser.add_argument('-t',       '--type',                type=int,      required=True, help='Types involved in experiment')
parser.add_argument("-V",       "--verbose",             type=int,      choices=[0,1])

args      =  parser.parse_args()
itern     =  int(args.itern if args.itern is not None else 0)
instance  =  int(args.siminst if args.siminst is not None else 0)
toplot    =  bool(args.toplot if args.toplot is not None else 0)
outdir    =  args.outdir if args.outdir is not None else 0

INDTYPE                       =  args.type
EXPERIMENT                    =  args.experiment if args.experiment is not None else "Unspecified"
VERBOSE                       =  True if args.verbose is not None and args.verbose !=0 else False
SHIFTING_FITNESS_PEAK         =  args.shifting_peak if args.shifting_peak is not None else False
CONNECTIVITY_FLAG             =  args.connectivity if args.connectivity is not None else False

MUTATION_RATE_ALLELE          =  0.01
MUTATION_VARIANTS_ALLELE      =  np.arange(-1,1,0.1)
MUTATION_RATE_DUPLICATION     =  0
MUTATION_RATE_CONTRIB_CHANGE  =  0
DEGREE                        =  1
COUNTER_RESET                 =  1024 * 8
STD                           =  1
AMPLITUDE                     =  1
LANDSCAPE_INCREMENT           =  0.5

INDIVIDUAL_INITS     =  {   
   "1":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ], dtype=np.float64)
   },
   "2":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ], dtype=np.float64) * np.array([-1,-1,1,1])
   },
   "3":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0],
                        [1,0],
                        [0,1],
                        [0,1],
                    ], dtype=np.float64) 
   },
   "4":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0],
                        [1,0],
                        [0,1],
                        [0,1],
                    ], dtype=np.float64) *np.array([1,1,-1,-1])[:,None]
   },
   "5":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0],
                        [1,0],
                        [0,1],
                        [0,1],
                    ], dtype=np.float64) *np.array([-1,1,-1,1])[:,None]
   },
   "6":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) * np.array([1,1,1,1])
   },
   "7":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) * np.array([1,1,-1,-1])
   },
   "8":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) * np.array([-1,1,-1,1])[:,None]
   },
   "9":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) * np.array([1,-1,1,-1])
   },
   "10":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,1,0,0],
                        [0,0,1,1],
                        [1,0,1,1],
                    ], dtype=np.float64) 
   },
   "11":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,1,0,0],
                        [0,0,-1,-1],
                        [1,0,-1,-1],
                    ], dtype=np.float64) 
   },
   "12":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [-1,-1, 0,  0],
                        [0, 1, 0,  0],
                        [0, 0,  -1, -1],
                        [1, 0,  1,  1],
                    ], dtype=np.float64) 
   },
   "13":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,1,1,0],
                        [0,0,1,1],
                        [1,0,0,1],
                    ], dtype=np.float64) 
   },
   "14":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,1,1,0],
                        [0,0,1,1],
                        [1,0,0,1],
                    ], dtype=np.float64) * np.array([1,1,-1,-1])
   },
   "15":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [-1,-1,0,0],
                        [0,1,-1,0],
                        [0,0,1,-1],
                        [1,0,0,1],
                    ], dtype=np.float64) 
   },
   "16":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                    ], dtype=np.float64) 
   },
   "17":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                    ], dtype=np.float64) * np.array([-1,-1,1,1])
   },
   "18":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                    ], dtype=np.float64) * np.array([-1,1,-1,1]) * np.array([-1,1,-1,1])[:,None]
   },
   "19":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ], dtype=np.float64)
   },
   "20":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ], dtype=np.float64) * np.array([-1,-1,1,1])
   },
   "21":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0],
                        [1,0],
                        [0,1],
                        [0,1],
                    ], dtype=np.float64) 
   },
   "22":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0],
                        [1,0],
                        [0,1],
                        [0,1],
                    ], dtype=np.float64) *np.array([1,1,-1,-1])[:,None]
   },
   "23":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0],
                        [1,0],
                        [0,1],
                        [0,1],
                    ], dtype=np.float64) *np.array([-1,1,-1,1])[:,None]
   },
   "24":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) * np.array([1,1,1,1])
   },
   "25":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) * np.array([1,1,-1,-1])
   },
   "26":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) * np.array([-1,1,-1,1])[:,None]
   },
   "27":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) * np.array([1,-1,1,-1])
   },
   "28":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,1,0,0],
                        [0,0,1,1],
                        [1,0,1,1],
                    ], dtype=np.float64) 
   },
   "29":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,1,0,0],
                        [0,0,-1,-1],
                        [1,0,-1,-1],
                    ], dtype=np.float64) 
   },
   "30":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [-1,-1, 0,  0],
                        [0, 1, 0,  0],
                        [0, 0,  -1, -1],
                        [1, 0,  1,  1],
                    ], dtype=np.float64) 
   },
   "31":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,1,1,0],
                        [0,0,1,1],
                        [1,0,0,1],
                    ], dtype=np.float64) 
   },
   "32":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,1,1,0],
                        [0,0,1,1],
                        [1,0,0,1],
                    ], dtype=np.float64) * np.array([1,1,-1,-1])
   },
   "33":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [-1,-1,0,0],
                        [0,1,-1,0],
                        [0,0,1,-1],
                        [1,0,0,1],
                    ], dtype=np.float64) 
   },
   "34":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                    ], dtype=np.float64) 
   },
   "35":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                    ], dtype=np.float64) * np.array([-1,-1,1,1])
   },
   "36":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                    ], dtype=np.float64) * np.array([-1,1,-1,1]) * np.array([-1,1,-1,1])[:,None]
   },
   "37":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,-1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) 
   },
   "38":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,-1,0,0],
                        [0,0,1,1],
                        [1,0,1,1],
                    ], dtype=np.float64) 
   },
   "39":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,-1,1,0],
                        [0,0,1,1],
                        [1,0,0,-1],
                    ], dtype=np.float64) 
   },
   "40":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,-1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) 
   },
   "41":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,-1,0,0],
                        [0,0,1,1],
                        [1,0,1,1],
                    ], dtype=np.float64) 
   },
   "42":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [0,-1,1,0],
                        [0,0,1,1],
                        [1,0,0,-1],
                    ], dtype=np.float64) 
   },
   "43":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,1],
                        [1,-1,1,1],
                        [-1,1,1,-1],
                        [1,1,-1,1],
                    ], dtype=np.float64) 
   },
}
class Fitmap():

    def __init__(self,std:float, amplitude:float, mean): 
        self.std                                       : float = std
        self.amplitude                                 : float = amplitude
        self.mean          = mean

    def getMap(self):
        def _(phenotype:np.ndarray):
            return             self.amplitude * math.exp(
                -(np.sum(((phenotype - self.mean)**2)/(2*self.std**2)))
                )
        return _

class GPMap():
    def __init__(self,contributions:np.ndarray) -> None:
        self.coeffs_mat     =  contributions

    def map_phenotype(self, alleles:np.ndarray  )->np.ndarray:
        return  np.sum(self.coeffs_mat * ( alleles ** 1), axis=1)

class Individual:

    def __init__(self, alleles:np.ndarray):
        self.alleles    =  alleles
        self.phenotype  =  []

    def give_birth(self)->Individual:

        def mutation_allele_cointoss(allele:float):
            return allele + ( np.random.choice([-1,1]) * 0.1)

        # def mutation_allele():
        #     return np.random.choice(MUTATION_VARIANTS_ALLELE)

        alleles_copy  =  np.copy(self.alleles)
        did_mutate = False
        for index, gene in enumerate( self.alleles.tolist() ):
            if np.random.uniform() <= MUTATION_RATE_ALLELE:
                alleles_copy[index]=mutation_allele_cointoss(gene)
                did_mutate = True

        if did_mutate:
            nascent = Individual(alleles_copy)
        else:
            nascent = Individual(self.alleles)
        return nascent

class Universe:

    def __init__ (self, initial_population:List[Individual], GPMap:GPMap,Fitmap:Fitmap) -> None:
        self.population  = []
        self.GPMap       = GPMap
        self.Fitmap      = Fitmap
        self.phenotypeHM = {}
        self.poplen      = 0
        self.iter        = 0
        self.avg_fitness = 0
        self.drate       = 0
        self.brate       = 0

        for i in initial_population:
            self.birth(i)

    def landscape_shift(self) ->None:
        for value in self.phenotypeHM:
            self.phenotypeHM[value]['f'] = self.Fitmap.getMap()(u.GPMap.map_phenotype(self.phenotypeHM[value]['a']))

    def serialize_phenhm(self):

        d2 = {
            'a':[],
            'f':0,
            'n':0
        }

        for k,e in self.phenotypeHM.items():
            d2[k] ={
                'a':e['a'].tolist(),
                'f':e['f'],
                'n':e['n']
            }

        return d2

    def hash_alleles(self,alleles:np.ndarray)->str:
        return xxhash.xxh64(str(alleles)).hexdigest()

    def get_avg_fitness(self):

        return reduce(lambda x,y: x + y['f']*y['n'] , self.phenotypeHM.values(),0)/self.poplen

    def tick(self)->None:
        self.iter         +=  1
        self.avg_fitness   =  self.get_avg_fitness()

        def pick_genotype():
            total_fitness =    reduce  (lambda t,h: t+h ,[*map(lambda x: x['n']*x['f'], self .phenotypeHM.values())])
            target_keys   = [* self                                                          .phenotypeHM.keys  () ]
            likelihoods   = [  genotype['n']*genotype['f']/total_fitness for genotype in self.phenotypeHM.values()]
            picked        =    np                                                            .random     .choice(target_keys,p=likelihoods)
            return self.phenotypeHM[picked]['a']

        chosen = np.random.choice(self.population)
        self.death(chosen)
        self.birth(Individual(pick_genotype()).give_birth())

    def death(self,_:Individual):

        i_hash = self.hash_alleles(_.alleles)
        self.phenotypeHM[i_hash]['n'] -= 1
        if self.phenotypeHM[i_hash]['n'] == 0:
            self.phenotypeHM.pop(i_hash)

        self.population.remove(_)
        self.poplen-=1

    def birth(self,individ:Individual)->None:

        K = self.hash_alleles(individ.alleles)

        if K in self.phenotypeHM:
            self.population.append(individ)
            self.phenotypeHM [K]['n']+=1
            self.poplen              +=1
        else:
            fitval = self.get_fitness(individ)
            self.population.append(individ)
            self.phenotypeHM[K] = {
                'a':np.round(individ.alleles,2),
                'f':fitval,
                'n':1
            }
            self.poplen+=1
            
    def get_fitness(self,ind:Individual) -> float:

        K                 = self.hash_alleles(ind.alleles)

        if K in self.phenotypeHM:
            return self.phenotypeHM[K]['f']
        else:
            phenotype         =  self.GPMap.map_phenotype(ind.alleles)
            fitval            =  self.Fitmap.getMap()(phenotype)
            return fitval

# ? --
# *-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
# ? --


count              =  []
fit                =  []
brate              =  []
if SHIFTING_FITNESS_PEAK:
    lsc  =  np.array([], ndmin=2)
mean        = np.array([0.0,0.0,0.0,0.0], dtype=np.float64)
ASYM_SWITCH = False
EXTINCTION  = False
gpm1 = GPMap      ( INDIVIDUAL_INITS[       str       (INDTYPE)]['coefficients'])
ftm  = Fitmap     ( STD              ,      AMPLITUDE, [0,0,0,0]                )
ind1 = Individual ( INDIVIDUAL_INITS[       str       (INDTYPE)]['alleles']     )
u    = Universe   ([ind1             ]*1000,gpm1      ,ftm                      )




for it in range(itern):

    if not it % 1000:
        count.append(u.poplen     )
        fit  .append(u.avg_fitness)
        brate.append(u.brate      )


    if SHIFTING_FITNESS_PEAK and not it % 1000:
        lsc = np.append(lsc, mean)

    if (not (it + 1 )  & (COUNTER_RESET -1 ) ) and SHIFTING_FITNESS_PEAK:        

        if SHIFTING_FITNESS_PEAK == 1:
            if np.max(mean) > 0.9:
                LANDSCAPE_INCREMENT    =  -0.5
                mean += LANDSCAPE_INCREMENT
            elif np.max(mean) < -0.9:
                LANDSCAPE_INCREMENT    =  0.5
                mean += LANDSCAPE_INCREMENT
            else:
                coin      = np.random.choice([-1,1])
                mean[0:] += coin*LANDSCAPE_INCREMENT
        elif SHIFTING_FITNESS_PEAK == -1:
            for i,x in enumerate(mean):
                if abs(x) == 1:
                    mean[i] += -mean[i]/2
                else:
                    mean[i] += np.random.choice([LANDSCAPE_INCREMENT,-LANDSCAPE_INCREMENT])

        u.Fitmap.mean=mean
        u.landscape_shift()
    u.tick()

exp = "exp{}".format(INDTYPE)
if outdir:
    lsc = np.reshape(lsc, (-1,4))
    data = pd.DataFrame({
        f"{INDTYPE}": count,
          "fit"     : fit,
          "brate"   : brate,
          "mean0"   : lsc[:,0],
          "mean1"   : lsc[:,1],
          "mean2"   : lsc[:,2],
          "mean3"   : lsc[:,3],

    })

    [count,fit,brate]=[*map(lambda x: np.around(x,5), [count,fit,brate])]
    os.makedirs(os.path.join(outdir,exp), exist_ok=True)
    data.to_parquet(os.path.join(outdir,exp,f'data{instance}.parquet'))
    with open(os.path.join(outdir,exp, f'phenotypeHM_{instance}.json'),'w') as outfile:
        json.dump(
            u.serialize_phenhm(),
            outfile)


pprint(u.phenotypeHM)