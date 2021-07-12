from pprint import pprint
import timeit
from datetime import datetime
from functools import reduce 
import json
from time import time
import xxhash
import sys, os
import numpy as np
import math
import argparse
import pickle

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

# parser .add_argument ("-it"       , "--itern"               , type= int      ,                 help = "The number of iterations"                                                                                            )
parser = argparse.ArgumentParser (                                          description           =                             'Simulation presets'                                                                                                                                          )
parser .add_argument ('-save'     , '--outdir'                  , type   = dir_path     ,                                help = ""                                                                                           "Specify the path to write the results of the simulation.""" )
parser .add_argument ("-itstart"  , "--iter_start"              , type   = int          ,required =True,                 help = "The number of iterations"                                                                                                                                )
parser .add_argument ("-itend"    , "--iter_end"                , type   = int          ,required =True,                 help = "The number of iterations"                                                                                                                                )
parser .add_argument ("-ls"       , "--landscape_increment"     , type   = float        ,required =True,                 help = "Simulation tag for the current instance."                                                                                                                )
parser .add_argument ("-sim"      , "--siminst"                 , type   = int          ,                                help = "Simulation tag for the current instance."                                                                                                                )
parser .add_argument ("-SP"       , "--shifting_peak"           , type   = int          , required=True,choices =[-1,1], help = "Flag for whether the fitness landscape changes or not."                                                                                                  )
parser .add_argument ('-t'        , '--type'                    , type   = int          ,required =True,                 help = 'Types involved in experiment'                                                                                                                            )
parser .add_argument ('-initn'    , '--initial_number'          , type   = int          ,                                help = 'Starting number of individuals'                                                                                                                          )
parser .add_argument ('-gpm_rate' , '--gpmrate'                 , type   = float        ,                                help = 'GP-map contribution change mutation rate'                                                                                                                )
parser .add_argument ('-alm_rate' , '--almrate'                 , type   = float        ,                                help = 'Allelic mutation rate'                                                                                                                                   )
parser .add_argument ('-re'       , '--resurrect'               , type   = dir_path     ,                                help = 'Path to reinstate the population from.'                                                                                                                  )
# parser .add_argument ('-logvar'   , '--log_variance_covariance' , action = 'store_true' ,                                help = 'Whether to collect variance and covariance values for the last tenth of the replicate run.'                                                              )

args                         = parser           .parse_args()
ITSTART                      = int(args.iter_start)
ITEND                        = int(args.iter_end)
INSTANCE_N                   = int       (args  .siminst    if args.siminst is not None else 0)
OUTDIR                       = args.outdir    if args.outdir     is not None else 0
RESSURECT_PATH               = args.resurrect if args.resurrect  is not None else 0
# LOG_VAR_COVAR_ON             = bool( args.log_variance_covariance )
INDTYPE                      = args.type
EXP                          = "exp{}".format(INDTYPE)
POPN                         = args.initial_number if args.initial_number is not None else 1000
SHIFTING_FITNESS_PEAK        = args.shifting_peak
LANDSCAPE_INCREMENT          = float(args.landscape_increment)
MUTATION_RATE_ALLELE         = 0.01 if args.almrate is None else float(args.almrate  )
MUTATION_RATE_CONTRIB_CHANGE = 0.01 if args.gpmrate is None else float( args.gpmrate )
DEGREE                       = 1
COUNTER_RESET                = 10000
STD                          = 1
AMPLITUDE                    = 1
LOG_FIT_EVERY                = int(1e5) #* every 100 generations
LOG_VAR_EVERY                = int(1e3) #* generation
PICKING_MEAN_STD             = ( 0,0.5 )

BEGIN_DATE = datetime.now().strftime("%I:%M%p on %B %d, %Y")

def print_receipt()->None:
    receipt = {
        "ITSTART"                     : ITSTART,
        "ITEND"                       : ITEND,
        "INSTANCE_N"                  : INSTANCE_N,
        "OUTDIR"                      : OUTDIR,
        "RESSURECT_PATH"              : RESSURECT_PATH,
        # "LOG_VAR_COVAR_ON"            : LOG_VAR_COVAR_ON,
        "INDTYPE"                     : INDTYPE,
        "EXP"                         : EXP,
        "POPN"                        : POPN,
        "SHIFTING_FITNESS_PEAK"       : SHIFTING_FITNESS_PEAK,
        "LANDSCAPE_INCREMENT"         : LANDSCAPE_INCREMENT,
        "MUTATION_RATE_ALLELE"        : MUTATION_RATE_ALLELE,
        "MUTATION_RATE_CONTRIB_CHANGE": MUTATION_RATE_CONTRIB_CHANGE,
        "DEGREE"                      : DEGREE,
        "COUNTER_RESET"               : COUNTER_RESET,
        "STD"                         : STD,
        "AMPLITUDE"                   : AMPLITUDE,
        "LOG_FIT_EVERY"               : LOG_FIT_EVERY,
        "LOG_VAR_EVERY"               : LOG_VAR_EVERY,
        "date_finished"               : datetime.now().strftime("%I:%M%p on %B %d, %Y"),
        "date_started"                : BEGIN_DATE,
        "PICKING_MEAN_STD" : [*PICKING_MEAN_STD]
    }
    with open(os.path.join(OUTDIR, "parameters_instance{}.json".format(INSTANCE_N)),'w') as infile:
        json.dump(receipt, infile)

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
}

[ os.makedirs(os.path.join(OUTDIR, intern_path), exist_ok=True) for intern_path in ['var_covar','fitness_data', 'terminal']]

class Fitmap():
    def __init__(self,std:float, amplitude:float, mean:np.ndarray)->None: 
        self.std       = std
        self.amplitude = amplitude
        self.mean      = mean

    def getMap(self):
        def _(phenotype:np.ndarray):
            return             self.amplitude * math.exp(
                -(np.sum(
                    ((phenotype - self.mean)**2)
                    /
                    (2*self.std**2)
                    )

                )
                )
        return _

class GPMap():

    def __init__(self,contributions:np.ndarray) -> None:
        self.coeffs_mat = np.array(contributions,dtype=np.float64)
        self.n_genes    = contributions.shape[1]

    def mutation_gpmap_contributions(self)->None:

        template   = np    .random.normal(*PICKING_MEAN_STD,(4,4))
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

    def __init__(self, alleles, gpmap ):
        self.alleles   = alleles
        self.gpmap     = gpmap
        self.phenotype = gpmap.map_phenotype(alleles)
        
    def give_birth(self):

        def mutation_allele_cointoss(allele):
            pick = np.random.normal(*PICKING_MEAN_STD, 1)
            return allele +  pick
        
        alleles_copy = np.copy(self.alleles)
        for index, gene in enumerate( self.alleles.tolist() ):
            if np.random.uniform() <= MUTATION_RATE_ALLELE:
                alleles_copy[index] = mutation_allele_cointoss(gene)

        newmap =  GPMap(self.gpmap.get_contributions())
        newmap.mutation_gpmap_contributions()
        nascent = Individual(alleles_copy,newmap)
        return nascent

class Universe:

    def __init__ (self, initial_population, Fitmap) :
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

    def aggregate_gpmaps(self):

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

    def landscape_shift(self) :
        #? For every class of phenotype, recalculate the fitness after the landscape has shifted.
        #? Individuals inside the given class are guaranteed to have the same phenotype, hence the same fitness.
        for hash_key in self.phenotypeHM:
            self.phenotypeHM[hash_key]['fitness'] = self.Fitmap.getMap()( self.phenotypeHM[hash_key]['phenotype'] )

    def hash_phenotype(self,P:np.ndarray)->str:
        return xxhash.xxh64(str(P)).hexdigest()

    def get_avg_fitness(self):
        return  reduce(lambda x,y: x + y['fitness']*y['n'] , self.phenotypeHM.values(),0)/self.poplen 

    def tick(self):
        self.iter         +=  1

        def pick_death():

            target_keys = [* self.phenotypeHM.keys  () ]
            likelihoods = [  phenotype['n']/self.poplen for phenotype in self.phenotypeHM.values()]
            picked_bin  = np.random.choice(target_keys,p=likelihoods)
            return (picked_bin, np.random.choice(self.phenotypeHM[picked_bin]['individuals']) )

        def pick_parent():

            total_fitness = reduce  (lambda t,h: t+h ,[*map(lambda x: x['n']*x['fitness'], self .phenotypeHM.values())])
            target_keys   = [* self                                                          .phenotypeHM.keys  () ]
            likelihoods   = [  phenotype['n']* phenotype['fitness']/total_fitness for phenotype in self.phenotypeHM.values()]
            picked_bucket = np.random.choice(target_keys,p=likelihoods)

            return np.random.choice(self.phenotypeHM[picked_bucket]['individuals'])

        self.death(*pick_death())
        self.birth(pick_parent().give_birth())

    def death(self,type_key,_):

        self.phenotypeHM[type_key]['individuals'].remove(_)
        self.phenotypeHM[type_key]['n'] -= 1
        self.poplen-=1

        if self.phenotypeHM[type_key]['n'] == 0:
            self.phenotypeHM.pop(type_key)

    def birth(self,_):

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
            
    def get_var_covar(self):

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

    def get_fitness(self,ind) :
        K                 = self.hash_phenotype(ind.phenotype)
        if K in self.phenotypeHM:
            return self.phenotypeHM[K]['fitness']
        else:
            return self.Fitmap.getMap()(ind.phenotype)

    def dump_state(self, output_directory):

        terminalpath = os.path.join(output_directory,'terminal')

        indout   = os.path.join(terminalpath, 'individuals_{}.json'.format(INSTANCE_N))
        indn     = 1
        ind_dump = {
            'fitness'  : self.Fitmap.mean.tolist(),
            'population': {}
        }
        for phen in self.phenotypeHM:
            ind_dump['population'][phen] ={}

            for individual in self.phenotypeHM[phen]['individuals']:
                individual
                ind_dump['population'][phen][indn] = {
                    "alleles": individual.alleles.tolist(),
                    "gpmap"  : individual.gpmap.coeffs_mat.tolist()
                }
                indn+=1

        with open(indout, 'w') as out:
            json.dump(ind_dump,out,sort_keys=True, indent=4)

    def log_var_covar(self, itern):
        var               , covar              = tuple(map( lambda _: np.around(_,5),self.get_var_covar()))
        self.var_covar_agg['var']                                   = np.sum([ self.var_covar_agg['var'], var ],axis=0)
        self.var_covar_agg['covar']                                 = np.sum([ self.var_covar_agg['covar'], covar ],axis=0)
        self.var_covar_agg['elapsed']                              += 1

        if                 self.var_covar_agg['began_loggin_at'] == -1:
            self.var_covar_agg['began_loggin_at']                       = itern

    def write_mean_var_covar(self,outdir):
        outfile    = os.path.join(outdir,'var_covar','mean_var_covar_{}.json'.format(INSTANCE_N))
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
    exp_number     ,
    instance_number,
    # * root path the the experiment (with /terminal,/fitness_data, etc..)
    inpath         ):

    pop_re = []
    try: 
        dump_f = os.path.join(inpath,'terminal','individuals_{}.json'.format(instance_number))
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
        print("Failed to open this combination of parameters when resurrecting a population:  exp {}  instance {} inpath {}.\n Exiting.".format(exp_number, INSTANCE_N, inpath))
        exit(1)

initial_landscape = [0,0,0,0]
fitmean_agg       = np.array([])

if RESSURECT_PATH:

    population, mean = ressurect_population(INDTYPE   , INSTANCE_N, RESSURECT_PATH)
    fitmap= Fitmap(STD,AMPLITUDE, mean)
    u= Universe(population,fitmap)

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

        cur_it      = np.array([u.get_avg_fitness(), *np.around(u.Fitmap.mean,2)])
        fitmean_agg = np.array([*fitmean_agg, cur_it])

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

        u.Fitmap.mean = mean
        u.landscape_shift()

    u.tick()

if OUTDIR:
    with open(os.path.join(OUTDIR,'fitness_data','data{}.csv'.format(INSTANCE_N)),'wb') as f: pickle.dump(fitmean_agg, f)
    u.write_mean_var_covar(OUTDIR)
    u.dump_state(OUTDIR)

print_receipt()


