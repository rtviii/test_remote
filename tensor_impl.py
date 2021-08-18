import random
import timeit
from datetime import datetime
from functools import reduce 
import json
from time import time
import sys, os
from typing import Callable, List
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

parser = argparse.ArgumentParser (                                          description           =                             'Simulation presets'                                                                                                                                          )
parser .add_argument ("-it"       , "--itern"               , type= int      ,                 help = "The number of iterations"                                                                                            )
parser .add_argument ('-save'     , '--outdir'                  , type   = dir_path     ,                                help = ""                                                                                           "Specify the path to write the results of the simulation.""" )
parser .add_argument ("-itstart"  , "--iter_start"              , type   = int          ,required =True,                 help = "The number of iterations"                                                                                                                                )
parser .add_argument ("-itend"    , "--iter_end"                , type   = int          ,required =True,                 help = "The number of iterations"                                                                                                                                )
parser .add_argument ("-ls"       , "--landscape_increment"     , type   = float        ,required =True,                 help = "Simulation tag for the current instance."                                                                                                                )
parser .add_argument ("-sim"      , "--siminst"                 , type   = int          ,                                help = "Simulation tag for the current instance."                                                                                                                )
parser .add_argument ("-SP"       , "--shifting_peak"           , type   = int          ,required=True,choices =[0,1], help = "Flag for whether the fitness landscape changes or not."                                                                                                  )
parser .add_argument ('-t'        , '--type'                    , type   = int          ,required =True,                 help = 'Types involved in experiment'                                                                                                                            )
parser .add_argument ('-initn'    , '--initial_number'          , type   = int          ,                                help = 'Starting number of individuals'                                                                                                                          )
parser .add_argument ('-gpm_rate' , '--gpmrate'                 , type   = float        ,                                help = 'GP-map contribution change mutation rate'                                                                                                                )
parser .add_argument ('-alm_rate' , '--almrate'                 , type   = float        ,                                help = 'Allelic mutation rate'                                                                                                                                   )
parser .add_argument ('-re'       , '--resurrect'               , type   = dir_path     ,                                help = 'Path to reinstate the population from.'                                                                                                                  )
# parser .add_argument ('-logvar'   , '--log_variance_covariance' , action = 'store_true' ,                                help = 'Whether to collect variance and covariance values for the last tenth of the replicate run.'                                                              )

args                         = parser .parse_args()
ITSTART                      = int(args.iter_start)
ITEND                        = int(args.iter_end)
REPLICATE_N                  = int (args .siminst if args.siminst is not None else 0)
OUTDIR                       = args.outdir if args.outdir is not None else 0
RESSURECT_PATH               = args.resurrect if args.resurrect is not None else 0
INDTYPE                      = args.type
POPN                         = args.initial_number if args.initial_number is not None else 1000
SHIFTING_FITNESS_PEAK        = args.shifting_peak
LS_INCREMENT                 = float(args.landscape_increment)
MUTATION_RATE_ALLELE         = 0.005 if args.almrate is None else float(args.almrate )
MUTATION_RATE_CONTRIB_CHANGE = 0.0001 if args.gpmrate is None else float( args.gpmrate )
DEGREE                       = 1
LS_SHIFT_EVERY               = int(1e4)
STD                          = 1
AMPLITUDE                    = 1
LOG_FIT_EVERY                = int(1e3) # * every 100 generations
LOG_VAR_EVERY                = int(1e3) # * generation
PICKING_MEAN_STD             = ( 0,0.5 )
BEGIN_DATE                   = datetime.now().strftime("%I:%M%p on %B %d, %Y")

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
#    testtype
   "0":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,1,-1,-1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1],
                    ], dtype=np.float64)
   },
#    testtype2
   "99":{
        'trait_n'       :  4,
        'alleles'       :  np.array([1,-1,-1,-1], dtype=np.float64),
        'coefficients'  :  np.array([
                        [0,1,1,1],
                        [0,1,1,1],
                        [0,1,1,1],
                        [0,1,1,1],
                    ], dtype=np.float64)
   },
}

[ os.makedirs(os.path.join(OUTDIR, intern_path), exist_ok=True) for intern_path in ['var_covar','fitness_data']]



class FitnessMap:

	def __init__(self, std):
		self.std = std

	def getmap(self, mean):

		u   = mean
		exp = math.exp

		def _(phenotype:np.ndarray):
			return AMPLITUDE * exp(-(np.sum(((phenotype - u)**2)/(2*self.std**2))))
		return _

def mutate_gpmap(contributions):

	probs           = np    .random.uniform(low=0, high=1, size=(4,4)).round(4)
	rows     , cols = probs .shape

	for i in  range(rows):
		for j in range(cols):
			if probs[i,j] <= MUTATION_RATE_CONTRIB_CHANGE:
				pick = np.random.normal(*PICKING_MEAN_STD, 1)
				contributions[i,j] += pick

def mutate_alleles(alleles:np.ndarray)->None:
	for g in range(alleles.shape[0]):
		if np.random.uniform() <= MUTATION_RATE_ALLELE:
			pick = np.random.normal(*PICKING_MEAN_STD, 1)
			alleles[g] += pick
	
def print_receipt()->None:
	receipt = {
		  "ITSTART"                     : ITSTART,
		  "ITEND"                       : ITEND,
		  "REPLICATE_N"                 : REPLICATE_N,
		  "OUTDIR"                      : OUTDIR,
		  "RESSURECT_PATH"              : RESSURECT_PATH,
		  "INDTYPE"                     : INDTYPE,
		  "POPN"                        : POPN,
		  "SHIFTING_FITNESS_PEAK"       : SHIFTING_FITNESS_PEAK,
		  "LS_INCREMENT"                : LS_INCREMENT,
		  "MUTATION_RATE_ALLELE"        : MUTATION_RATE_ALLELE,
		  "MUTATION_RATE_CONTRIB_CHANGE": MUTATION_RATE_CONTRIB_CHANGE,
		  "DEGREE"                      : DEGREE,
		  "LS_SHIFT_EVERY"              : LS_SHIFT_EVERY,
		  "STD"                         : STD,
		  "AMPLITUDE"                   : AMPLITUDE,
		  "LOG_FIT_EVERY"               : LOG_FIT_EVERY,
		  "LOG_VAR_EVERY"               : LOG_VAR_EVERY,
		  "date_finished"               : datetime                    .now().strftime("%I:%M%p on %B %d, %Y"),
		  "date_started"                : BEGIN_DATE,
		  "PICKING_MEAN_STD"            : [*PICKING_MEAN_STD]
	}
	with open(os.path.join(OUTDIR, "parameters_replicate{}.json".format(REPLICATE_N)),'w') as infile:
		json.dump(receipt, infile)

class Universe:

	def __init__(
		self,
		current_iter:int,
		ALLS: np.ndarray,
		GPMS: np.ndarray,
		PHNS: np.ndarray,
		fmap: FitnessMap,
		mean: np.ndarray,
		) -> None:

		self.it     = current_iter
		self.ALLS   = ALLS
		self.GPMS   = GPMS
		self.PHNS   = PHNS
		self.fitmap = fmap
		self.mean = mean
		# ? ------------------------------ AGGREGATORS
		self.var_covar_agg = {
			"began_loggin_at": -1,
			"logged_every"   : LOG_VAR_EVERY,
			'phenotype_agg' : [],
			"elapsed"        : 0
		}
		self.fitmean_agg   = np.array([])
		self.mean_phen_agg = np.mean(self.PHNS, axis=0)

	def pick_parent(self)->int:

		indices   = np.arange(len( self.PHNS ))
		fitnesses = np.array( [  *map( Fitmap.getmap(self.mean), self.PHNS)] )
		cumfit    = reduce(lambda x,y : x+y, fitnesses)

		return np.random.choice(indices,p=fitnesses/cumfit)

	def pick_death(self)->int:
		indices   = np.arange(len( self.PHNS ))
		return np.random.choice(indices)

	def tick(self):

		self.it += 1

		if self.it > ITEND-(math.ceil((ITEND-ITSTART)/10)) and not(self.it%(LOG_VAR_EVERY)):

			self.log_var_covar()

		if not self.it % LOG_FIT_EVERY:
			cur_it           = np.array([self.get_avg_fitness(), *np.around(self.mean,2)])
			self.fitmean_agg = np.array([*self.fitmean_agg, cur_it])


		if ( not self.it % LS_SHIFT_EVERY ):
			self.shift_landscape(LS_INCREMENT,SHIFTING_FITNESS_PEAK)


		death_index = self.pick_death()
		birth_index = self.pick_parent()

		_alleles    = np.copy(self.ALLS[birth_index])
		_contribs   = np.copy(self.GPMS[birth_index])

		mutate_alleles(_alleles)
		mutate_gpmap(_contribs)

		self.PHNS[death_index] =  _contribs @ _alleles.T
		self.ALLS[death_index] = _alleles
		self.GPMS[death_index] = _contribs

	def get_avg_fitness(self)->float:
		return reduce(lambda x,y: x + y, map(self.fitmap.getmap(self.mean), self.PHNS))/len(self.PHNS)

	def write_covar_pkl(self,outdir):

		outfile    = os.path.join(outdir,'var_covar','mean_var_covar_{}.pkl'.format(REPLICATE_N))
		_ = {
			'phenotype_agg'   : self.mean_phen_agg,
			'elapsed'         : self.var_covar_agg[ 'elapsed'         ],
			'began_loggin_at' : self.var_covar_agg[ 'began_loggin_at' ],
			'logged_every'    : self.var_covar_agg[ 'logged_every'    ]
		}
		with open(outfile,'wb') as log:
			pickle.dump(_, log)

	def log_var_covar(self):
		self.mean_phen_agg = np.row_stack([ self.mean_phen_agg, np.mean(self.PHNS, axis=0) ])

		self.var_covar_agg['elapsed']                              += 1
		if                 self.var_covar_agg['began_loggin_at'] == -1:
			self.var_covar_agg['began_loggin_at']                       = self.it

	def shift_landscape(
	 	self,
	 	LANDSCAPE_INCREMENT: float,
	 	CORRELATED         : bool)->None  : 

	#? Corellated shifts
	 	if CORRELATED:
	 		# *Large IT
	 		if abs(LANDSCAPE_INCREMENT) > 0.9:
	 			if np.max(self.mean) > 0.9:
	 				if np.random.choice([-1,1]) > 0:
	 					self.mean -= LANDSCAPE_INCREMENT
	 			elif np.min(self.mean) < -0.9:
	 				if np.random.choice([-1,1]) > 0:
	 					self.mean += LANDSCAPE_INCREMENT
	 			else:
	 				self.mean +=  np.random.choice([-1,1]) 
	 		# *Small IT
	 		else:
	 			if np.max(self.mean) > 0.9:
	 				if np.random.choice([1,-1]) >0:
	 					self.mean -= LANDSCAPE_INCREMENT
	 				else:
	 					...
	 			elif np.min(self.mean) < -0.9:
	 				if np.random.choice([1,-1]) >0:
	 					self.mean += LANDSCAPE_INCREMENT
	 				else:
	 					...
	 			else: 
	 				self.mean += np.random.choice([1,-1])*LANDSCAPE_INCREMENT
	 	#? Uncorellated shifts
	 	else:
	 		# *Large IT
	 		if abs(LANDSCAPE_INCREMENT) > 0.9:
	 			for i,x in enumerate(self.mean):
	 				if self.mean[i] > 0.9:
	 					if np.random.choice([1,-1]) >0:
	 						self.mean[i] -= LANDSCAPE_INCREMENT
	 					else:
	 						...
	 				elif self.mean[i] < -0.9:
	 					if np.random.choice([1,-1]) >0:
	 						self.mean[i] += LANDSCAPE_INCREMENT
	 					else:
	 						...

	 				else:
	 					self.mean[i]+=  np.random.choice([1,-1]) * LANDSCAPE_INCREMENT
	 		# *Small IT
	 		else:
	 			for i,x in enumerate(self.mean):
	 				if self.mean[i] > 0.9:
	 					if np.random.choice([1,-1]) >0:
	 						self.mean[i] -= abs(LANDSCAPE_INCREMENT)
	 					else:
	 						...
	 				elif self.mean[i] < -0.9:
	 					if np.random.choice([1,-1]) >0:
	 						self.mean += LANDSCAPE_INCREMENT
	 					else:
	 						...
	 				else: 
	 					coin = np.random.choice([1,-1])
	 					self.mean[i] += coin*LANDSCAPE_INCREMENT


#***************** INITS ***************

alls     = np        .array([ INDIVIDUAL_INITS[str( INDTYPE )]['alleles' ] for i in range(POPN) ], dtype=np.float64)
gpms     = np        .array([ INDIVIDUAL_INITS[str( INDTYPE )]['coefficients'] for i in range(POPN)	], dtype=np.float64)
phns     = np        .array( [gpms[i]@ alls[i].T for i in range(POPN) ], dtype=np.float64)
Fitmap   = FitnessMap      (1)
universe = Universe        (ITSTART,
                    alls,
                    gpms,
                    phns,
                    Fitmap,
                    np.array([0,0,0,0], dtype=np.float64))

#*******************************************************Y

#!  -  +  -  + -  +  -  + 
for it in range(ITSTART, ITEND+1): universe.tick()
#!  -  +  -  + -  +  -  + 

if OUTDIR:
    with open(os.path.join(OUTDIR,'fitness_data','data{}.pkl'.format(REPLICATE_N)),'wb') as f: pickle.dump(universe.fitmean_agg, f)
    universe.write_covar_pkl(OUTDIR)
print_receipt()

