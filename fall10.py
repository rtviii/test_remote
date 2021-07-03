from __future__ import annotations
from functools import reduce 
import functools
import json
import xxhash
import csv
import sys, os
import numpy as np
from typing import   List, Tuple
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

parser = argparse.ArgumentParser (                             description        = 'Simulation presets'                                                             )
parser           .add_argument   ('-save' , '--outdir' , type= dir_path    , help = ""                  "Specify the path to write the results of the simulation.""" )
# parser .add_argument ("-it"       , "--itern"               , type= int      ,                 help = "The number of iterations"                                                                                            )
parser .add_argument ("-itstart"  , "--iter_start"              , type  = int        ,required=True,   help = "The number of iterations"                                                                   )
parser .add_argument ("-itend"    , "--iter_end"                , type  = int        ,required=True,   help = "The number of iterations"                                                                   )
parser .add_argument ("-ls"       , "--landscape_increment"     , type  = float      ,required=True,   help = "Simulation tag for the current instance."                                                   )
parser .add_argument ("-sim"      , "--siminst"                 , type  = int        ,                 help = "Simulation tag for the current instance."                                                   )
parser .add_argument ("-SP"       , "--shifting_peak"           , type  = int        ,choices =[-1,1], help = "Flag for whether the fitness landscape changes or not."                                     )
parser .add_argument ('-t'        , '--type'                    , type  = int        ,required=True,   help = 'Types involved in experiment'                                                               )
parser .add_argument ('-initn'    , '--initial_number'          , type  = int        ,                 help = 'Starting number of individuals'                                                             )
parser .add_argument ('-gpm_rate' , '--gpmrate'                 , type  = float      ,                 help = 'GP-map contribution change mutation rate'                                                   )
parser .add_argument ('-alm_rate' , '--almrate'                 , type  = float      ,                 help = 'Allelic mutation rate'                                                                      )
parser .add_argument ('-re'       , '--resurrect'               , type  = dir_path   ,                 help = 'Path to reinstate the population from.'                                                     )
parser .add_argument ('-logvar'   , '--log_variance_covariance' , action='store_true',                 help = 'Whether to collect variance and covariance values for the last tenth of the replicate run.' )

args                         = parser           .parse_args()
ITSTART                      = int(args.iter_start)
ITEND                        = int(args.iter_end)
INSTANCE_N                   = int       (args  .siminst    if args.siminst is not None else 0)
OUTDIR                       = args.outdir    if args.outdir     is not None else 0
RESSURECT_PATH               = args.resurrect if args.resurrect  is not None else 0
LOG_VAR_COVAR_ON             = bool( args.log_variance_covariance )
INDTYPE                      = args.type
EXP                          = "exp{}".format(INDTYPE)
POPN                         = args.initial_number if args.initial_number is not None else 1000
SHIFTING_FITNESS_PEAK        = args.shifting_peak if args.shifting_peak is not None else False
LANDSCAPE_INCREMENT          = float(args.landscape_increment)
MUTATION_RATE_ALLELE         = 0.00016 if args.almrate is None else float(args.almrate  )
MUTATION_RATE_CONTRIB_CHANGE = 0.00004 if args.gpmrate is None else float( args.gpmrate )
MUTATION_VARIANTS_ALLELE     = np.arange(-1,1,0.1)
DEGREE                       = 1
COUNTER_RESET                = 500000
STD                          = 1
AMPLITUDE                    = 1
LOG_FIT_EVERY                = int(1e3)
LOG_VAR_EVERY                = int(1e3)

def print_receipt(self)->None:
    receipt = {

    }

#? Create a parameter log file to write out.
INDIVIDUAL_INITS     =  {   
#    mendelian
   "1":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ])
   },
#    modularpositive
   "2":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) 
   },
#    modularskipping
   "3":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,-1,0,0],
                        [0,0,1,1],
                        [0,0,1,-1]])
   },
#    spiderweb
   "4":{

        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                    ], dtype=np.float64)
   },
#    mendelian
   "5":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ])
   },
#    modularpositive
   "6":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) 
   },
#    modularskipping
   "7":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,-1,0,0],
                        [0,0,1,1],
                        [0,0,1,-1]])
   },
#    spiderweb
   "8":{

        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                    ], dtype=np.float64)
   },
#    mendelian
   "9":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ])
   },
#    modularpositive
   "10":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) 
   },
#    modularskipping
   "11":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,-1,0,0],
                        [0,0,1,1],
                        [0,0,1,-1]])
   },
#    spiderweb
   "12":{

        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,1,1,1],
                    ], dtype=np.float64)
   },
#    mendelian
   "13":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ])
   },
#    modularpositive
   "14":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) 
   },
#    modularskipping
   "15":{
        'trait_n' :4,
        'alleles'       :  np.array([1,1,1,1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,1,0,0],
                        [1,-1,0,0],
                        [0,0,1,1],
                        [0,0,1,-1]])
   },
#    spiderweb
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
}

[ os.makedirs(os.path.join(OUTDIR, intern_path), exist_ok=True) for intern_path in ['var_covar','end_phenotypes','fitness_data', 'terminal']]

class Fitmap():
    def __init__(self,std:float, amplitude:float, mean:np.ndarray)->None: 
        self.std                                       : float = std
        self.amplitude                                 : float = amplitude
        self.mean:np.ndarray = mean

    def getMap(self):
        # * Returns a Callable[...] but leaving the annotation out.
        def _(phenotype:np.ndarray):
            return             self.amplitude * math.exp(
                -(np.sum(((phenotype - self.mean)**2)/(2*self.std**2)))
                )
        return _

class GPMap():

    def __init__(self,contributions:np.ndarray) -> None:
        self.coeffs_mat = contributions
        self.n_genes    = contributions.shape[1]

    def mutation_gpmap_contributions(self)->None:
        template   = np    .random.randint(-1,2,(4,self.n_genes))
        probs      = np    .random.uniform(low=0, high=1, size=(4,self.n_genes)).round(4)
        rows , cols = probs .shape

        for i in  range(rows):
            for j in range(cols):
                if probs[i,j] <= MUTATION_RATE_CONTRIB_CHANGE:
                    self.coeffs_mat[i,j] = template[i,j]

    def get_contributions(self) ->np.ndarray:
        return np.copy( self.coeffs_mat )

    def map_phenotype(self, alleles:np.ndarray  )->np.ndarray:
        return  np.sum(self.coeffs_mat * ( alleles ** 1), axis=1)

class Individual:

    def __init__(self, alleles:np.ndarray, gpmap:GPMap )->None:
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

class Universe:

    def __init__ (self, initial_population:List[Individual], Fitmap:Fitmap) -> None:
        self.Fitmap                           = Fitmap
        # * a dictionary of phenotype
        self.phenotypeHM = {}
        # an aggregator for variance/covariance calculations
        self.var_covar_agg = {
            "log_var_covar"  : LOG_VAR_COVAR_ON,
            "began_loggin_at": -1,
            "logged_every"   : LOG_VAR_EVERY,
            "var"            : np.array([0,0,0,0], dtype=np.float64),
            "covar"          : np.array([0,0,0,0,0,0], dtype=np.float64),
            "elapsed": 0
        }
        self.poplen        = 0
        self.iter          = 0

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
            
    def get_var_covar(self)->List[np.ndarray]:

        traits = {
            0:np.array([]),
            1:np.array([]),
            2:np.array([]),
            3:np.array([]),
        }

        for phen in self.phenotypeHM.values():
           for index,trait in enumerate( phen['phenotype'] ):
               for x in range(phen['n']):
                   traits[index] = np.append(traits[index],trait)

        var = np.zeros(( 4 ))
        for _ in range(0,4):
            var[_] = np.var(traits[_])
         
        _traitsfull= np.array([
            traits[0],
            traits[1],
            traits[2],
            traits[3],
        ])

        cov = np.cov(_traitsfull, bias=True)
        t1t2 = cov[0,1]
        t1t3 = cov[0,2]
        t1t4 = cov[0,3]
        t2t3 = cov[1,2]
        t2t4 = cov[1,3]
        t3t4 = cov[2,2]

        return [np.array(var), np.array([t1t2,t1t3,t1t4,t2t3,t2t4,t3t4])]

    def get_fitness(self,ind:Individual) -> float:
        K                 = self.hash_phenotype(ind.phenotype)
        if K in self.phenotypeHM:
            return self.phenotypeHM[K]['fitness']
        else:
            return self.Fitmap.getMap()(ind.phenotype)

    def dump_state(self, output_directory:str)->None:

        terminalpath = os.path.join(output_directory,'terminal')

        indout   = os.path.join(terminalpath, f'individuals_{INSTANCE_N}.json')
        indn     = 1
        ind_dump = {
            'fitness'  : self.Fitmap.mean.tolist(),
            'population': {}
        }
        for phen in self.phenotypeHM:
            ind_dump['population'][phen] ={}

            for individual in self.phenotypeHM[phen]['individuals']:
                individual:Individual
                ind_dump['population'][phen][indn] = {
                    "alleles": individual.alleles.tolist(),
                    "gpmap"  : individual.gpmap.coeffs_mat.tolist()
                }
                indn+=1

        with open(indout, 'w') as out:
            json.dump(ind_dump,out,sort_keys=True, indent=4)

    def log_var_covar(self, itern:int)->None:
        var               , covar              = tuple(map( lambda _: np.around(_,5),self.get_var_covar()))

        self.var_covar_agg['var']                                   = np.sum([ self.var_covar_agg['var'], var ],axis=0)
        self.var_covar_agg['covar']                                 = np.sum([ self.var_covar_agg['covar'], covar ],axis=0)
        self.var_covar_agg['elapsed']                              += 1

        if                 self.var_covar_agg['began_loggin_at'] == -1:
            self.var_covar_agg['began_loggin_at']                       = itern

    def write_mean_var_covar(self,outdir:str) ->None:
        outfile    = os.path.join(outdir,'var_covar',f'mean_var_covar_{INSTANCE_N}.json')
        _ = {
            'var'            : self.var_covar_agg['var'    ].tolist(),
            'covar'          : self.var_covar_agg['covar'  ].tolist(),
            'elapsed'        : self.var_covar_agg['elapsed'],
            'began_loggin_at': self.var_covar_agg[ 'began_loggin_at' ],
            'logged_every'   : self.var_covar_agg[ 'logged_every' ]
        }
        with open(outfile,'w') as log:
            json.dump(_, log)

def ressurect_population(
    exp_number     : int,
    instance_number: int,
    # * root path the the experiment (with /terminal,/fitness_data, etc..)
    inpath         : str)->[List[Individual], np.ndarray]: 

    pop_re:List[Individual] = []
    try: 
        dump_f = os.path.join(inpath,'terminal',f'individuals_{instance_number}.json')
        with open(dump_f) as infile:
            data = json.load(infile)
        fitmean = np.array( data['fitness'] )
        pop  = data['population']
        for pheno in pop:
            phenotype = pop[pheno]
            for individual in phenotype:
                record = phenotype[individual]
                als    = np.array(record['alleles'])
                coefs  = np.array( record['gpmap'] )
                i      = Individual(als, GPMap(coefs))
                pop_re.append(i)

        return [ pop_re, fitmean ]
    except:
        print(f"""Failed to open this combination of parameters when resurrecting a population:  exp {exp_number}  instance {INSTANCE_N} inpath {inpath}.\n Exiting.""")
        exit(1)

count              =  []
fit                =  []

if SHIFTING_FITNESS_PEAK:
    lsc  =  np.array([], ndmin=2)

initial_landscape = [0,0,0,0]


if RESSURECT_PATH:
    population, mean = ressurect_population(INDTYPE   , INSTANCE_N, RESSURECT_PATH)
    fitmap:Fitmap= Fitmap(STD,AMPLITUDE, mean)
    u                  = Universe            (population,fitmap                   )
    
else:
    initial_landscape = np.array(initial_landscape, dtype=np.float64)
    mean              = np.array(initial_landscape,dtype=np.float64)
    ftm               = Fitmap( STD,AMPLITUDE, mean)
    init_population   = [ 
        Individual(INDIVIDUAL_INITS[str(INDTYPE)]['alleles'],
        GPMap(INDIVIDUAL_INITS[str(INDTYPE)]['coefficients'])) 
        for x in range(POPN)
        ]
    u = Universe(init_population,ftm)


for it in range(ITSTART, ITEND+1):
    if it > ITEND-(math.ceil((ITEND-ITSTART)/10)) and not(it%(LOG_VAR_EVERY)):
        u.log_var_covar(it)

    if not it % LOG_FIT_EVERY:
        fit.append(u.get_avg_fitness())
        if SHIFTING_FITNESS_PEAK!=0:
            lsc = np.append(lsc, mean)

    if ((not it%COUNTER_RESET) and SHIFTING_FITNESS_PEAK!=0):        
        #? Corellated shifts
        if SHIFTING_FITNESS_PEAK == 1:
            # *Large IT
            if abs(LANDSCAPE_INCREMENT) > 0.9:
                if np.max(mean) > 0.9:
                    if np.random.choice([-1,1]) > 0:
                        mean -= LANDSCAPE_INCREMENT
                elif np.min(mean) < -0.9:
                    if np.random.choice([-1,1]) > 0:
                        mean += LANDSCAPE_INCREMENT
                else:
                    mean +=  np.random.choice([-1,1]) 
            # *Small IT
            else:
                if np.max(mean) > 0.9:

                    if np.random.choice([1,-1]) >0:
                        mean -= LANDSCAPE_INCREMENT
                    else:
                        ...
                elif np.min(mean) < -0.9:
                    if np.random.choice([1,-1]) >0:
                        mean += LANDSCAPE_INCREMENT
                    else:
                        ...
                else: 
                    mean += np.random.choice([1,-1])*LANDSCAPE_INCREMENT
                
        #? Uncorellated shifts
        elif SHIFTING_FITNESS_PEAK == -1:
            # *Large IT
            if abs(LANDSCAPE_INCREMENT) > 0.9:
                for i,x in enumerate(mean):
                    if mean[i] > 0.9:
                        if np.random.choice([1,-1]) >0:
                            mean[i] -= LANDSCAPE_INCREMENT
                        else:
                            ...

                    elif mean[i] < -0.9:
                        if np.random.choice([1,-1]) >0:
                            mean[i] += LANDSCAPE_INCREMENT
                        else:
                            ...
                    else:
                        mean[i]+=  np.random.choice([1,-1]) * LANDSCAPE_INCREMENT


            # *Small IT
            else:
                for i,x in enumerate(mean):
                    if mean[i] > 0.9:
                        if np.random.choice([1,-1]) >0:
                            mean[i] -= abs(LANDSCAPE_INCREMENT)
                        else:
                            ...
                    elif mean[i] < -0.9:
                        if np.random.choice([1,-1]) >0:
                            mean += LANDSCAPE_INCREMENT
                        else:
                            ...
                    else: 
                        coin = np.random.choice([1,-1])
                        mean[i] += coin*LANDSCAPE_INCREMENT

        u.Fitmap.mean=mean
        u.landscape_shift()
    u.tick()


if OUTDIR:

    lsc  = np.reshape(lsc, (-1,4))
    data = pd.DataFrame({
        "fit"     : fit,
        "mean0"   : lsc[:,0],
        "mean1"   : lsc[:,1],
        "mean2"   : lsc[:,2],
        "mean3"   : lsc[:,3],
    })


    
    [count,fit]=[*map(lambda x: np.around(x,5), [count,fit])]

    data.to_parquet(os.path.join(OUTDIR,'fitness_data',f'data{INSTANCE_N}.parquet'))
    with open(os.path.join(OUTDIR,'end_phenotypes', f'gpmaps_{INSTANCE_N}.json'), 'w') as outfile:
        json.dump(u.aggregate_gpmaps(),outfile)

    u.write_mean_var_covar(OUTDIR)
    u.dump_state(OUTDIR)




