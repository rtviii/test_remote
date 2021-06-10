from __future__ import annotations
from ast import Call
from functools import reduce 
import functools
import json
from operator import xor
import xxhash
import csv
import sys, os
import numpy as np
from typing import   List 
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

MUTATION_RATE_ALLELE          =  0.0001
MUTATION_VARIANTS_ALLELE      =  np.arange(-1,1,0.1)
MUTATION_RATE_CONTRIB_CHANGE  =  0.0001
MUTATION_RATE_DUPLICATION     =  0
DEGREE                        =  1
COUNTER_RESET                 =  100000
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
        self.mean:np.ndarray = mean

    def getMap(self):
        def _(phenotype:np.ndarray):
            return             self.amplitude * math.exp(
                -(np.sum(((phenotype - self.mean)**2)
                /
                (2*self.std**2)))
                )
        return _

class GPMap():

    def __init__(self,contributions:np.ndarray) -> None:
        self.coeffs_mat = contributions
        self.n_genes    = contributions.shape[1]

    def mutation_gpmap_contributions(self)->None:
        template   = np    .random.randint(-1,2,(4,self.n_genes))
        probs      = np    .random.uniform(low=0, high=1, size=(4,self.n_genes)).round(4)
        rows ,cols = probs .shape

        for i in  range(rows):
            for j in range(cols):
                if probs[i,j] <= MUTATION_RATE_CONTRIB_CHANGE:
                    self.coeffs_mat[i,j] = template[i,j]

    def get_contributions(self) ->np.ndarray:
        return np.copy( self.coeffs_mat )

    def map_phenotype(self, alleles:np.ndarray  )->np.ndarray:
        return  np.sum(self.coeffs_mat * ( alleles ** 1), axis=1)

class Individual:

    def __init__(self, alleles:np.ndarray, gpmap:GPMap ):
        self.alleles   = alleles
        self.gpmap     = gpmap
        self.phenotype = gpmap.map_phenotype(alleles)
        
    def give_birth(self)->Individual:

        def mutation_allele_cointoss(allele:float):
            return allele + ( np.random.choice([-1,1]) * 0.1)
        
        alleles_copy = np.copy(self.alleles)
        for index, gene in enumerate( self.alleles.tolist() ):
            if np.random.uniform() <= MUTATION_RATE_ALLELE:
                alleles_copy[index] = mutation_allele_cointoss(gene)

        newmap =  GPMap(self.gpmap.get_contributions())
        newmap.mutation_gpmap_contributions()
        nascent = Individual(alleles_copy,newmap)
        return nascent

# class Phenotype(TypedDict): 
#       phenotype           : np.ndarray
#       fitness             : float
#       individuals         : List[Individual]
#       n                   : int

class Universe:

    def __init__ (self, initial_population:List[Individual], Fitmap:Fitmap) -> None:
        self.Fitmap                           = Fitmap
        # * a dictionary of phenotype
        self.phenotypeHM = {}
        self.poplen                           = 0
        self.iter                             = 0

        for i in initial_population:
            self.birth(i)

    def aggregate_gpmaps(self)->dict:

        _ = {}

        for ph in self.phenotypeHM:
            for index, ind in enumerate(self.phenotypeHM[ph]['individuals']):
                key = xxhash.xxh64(str(ind.gpmap.get_contributions())).hexdigest()
                if  key in _:
                    _[key]['n'] += 1

                else:
                    _[key] = {
                        "contributions": ind.gpmap.get_contributions().tolist(),
                        "n"            : 1
                    }
        return _

    def landscape_shift(self) ->None:
        #? For every class of phenotype, recalculate the fitness after the landscape has shifted.
        #? Individuals inside the given class are guaranteed to have the same phenotype, hence the same fitness.

        for hash_key in self.phenotypeHM:
            self.phenotypeHM[hash_key]['fitness'] = self.Fitmap.getMap()( self.phenotypeHM[hash_key]['phenotype'] )


    def hash_phenotype(self,P:np.ndarray)->str:
        return xxhash.xxh64(str(P)).hexdigest()

    def get_avg_fitness(self):

        return reduce(lambda x,y: x + y['fitness']*y['n'] , self.phenotypeHM.values(),0)/self.poplen

    def tick(self)->None:
        self.iter         +=  1

        def pick_death()->tuple[str, Individual]:

            target_keys = [* self.phenotypeHM.keys  () ]
            likelihoods = [  phenotype['n']/self.poplen for phenotype in self.phenotypeHM.values()]
            picked_bin  = np.random.choice(target_keys,p=likelihoods)
            return (picked_bin, np.random.choice(self.phenotypeHM[picked_bin]['individuals']) )

        def pick_parent()->Individual: 

            total_fitness = reduce  (lambda t,h: t+h ,[*map(lambda x: x['n']*x['fitness'], self .phenotypeHM.values())])
            target_keys   = [* self                                                          .phenotypeHM.keys  () ]
            likelihoods   = [  phenotype['n']* phenotype['fitness']/total_fitness for phenotype in self.phenotypeHM.values()]
            picked_bucket = np.random.choice(target_keys,p=likelihoods)

            return np.random.choice(self.phenotypeHM[picked_bucket]['individuals'])

        self.death(*pick_death())
        self.birth(pick_parent().give_birth())

    def death(self,type_key:str,_:Individual):

        self.phenotypeHM[type_key]['individuals'].remove(_)
        self.phenotypeHM[type_key]['n'] -= 1
        self.poplen-=1

        if self.phenotypeHM[type_key]['n'] == 0:
            self.phenotypeHM.pop(type_key)

    def birth(self,_:Individual)->None:

        K = self.hash_phenotype(_.phenotype)
        if K in self.phenotypeHM:
            self.phenotypeHM[K]['individuals'].append(_)
            self.phenotypeHM[K]['n']+=1
        else:
            self.phenotypeHM[K] = {
                'fitness'    : self.get_fitness(_),
                'n'          : 1,
                'phenotype'  : _.phenotype,
                'individuals': [_]
            }

        self.poplen+=1
            
    def get_fitness(self,ind:Individual) -> float:
        K                 = self.hash_phenotype(ind.phenotype)
        if K in self.phenotypeHM:
            return self.phenotypeHM[K]['fitness']
        else:
            return self.Fitmap.getMap()(ind.phenotype)

count              =  []
fit                =  []

if SHIFTING_FITNESS_PEAK:
    lsc  =  np.array([], ndmin=2)

mean            = np.array([0.0,0.0,0.0,0.0], dtype=np.float64)
ftm             = Fitmap( STD,AMPLITUDE, [0,0,0,0])
init_population = [ 
    Individual(INDIVIDUAL_INITS[str(INDTYPE)]['alleles'],
    GPMap(INDIVIDUAL_INITS[str(INDTYPE)]['coefficients'])) 
    for x in range(1000)
    ]

u = Universe(init_population,ftm)


for it in range(1,itern):

    if not it % 1000:
        fit.append(u.get_avg_fitness())

    if SHIFTING_FITNESS_PEAK and not it % 1000:
        lsc = np.append(lsc, mean)

    if ((not it%COUNTER_RESET) and SHIFTING_FITNESS_PEAK!=0):        

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

    lsc  = np.reshape(lsc, (-1,4))
    data = pd.DataFrame({
          "fit"     : fit,
          "mean0"   : lsc[:,0],
          "mean1"   : lsc[:,1],
          "mean2"   : lsc[:,2],
          "mean3"   : lsc[:,3],
    })

    [count,fit]=[*map(lambda x: np.around(x,5), [count,fit])]
    os.makedirs(os.path.join(outdir,exp), exist_ok=True)
    data.to_parquet(os.path.join(outdir,exp,f'data{instance}.parquet'))

    with open(os.path.join(outdir,exp, f'gpmaps_{instance}.json'), 'w') as outfile:
        json.dump(u.aggregate_gpmaps(),outfile)
