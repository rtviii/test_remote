#!/bin/bash

#PBS -l walltime=16:00:00,select=1:ncpus=1:mem=8gb
#PBS -J 0-100:1
#PBS -N gpmut_exp_17
#PBS -A ex-kdd-1
#PBS -m abe    
#PBS -M rtkushner@alumni.ubc.ca                                      
#PBS -o ./outputs/output.txt
#PBS -e ./outputs/error.txt
 
################################################################################
 
module load python3

cd $PBS_O_WORKDIR

source ../polyvenv/bin/activate


python3 ../staticpopv5.py -sim $PBS_ARRAY_INDEX -t 17 -it 5000000 -SP -1 --outdir  $PBS_O_WORKDIR 

