#!/bin/bash

#PBS -l walltime=12:00:00,select=1:ncpus=1:mem=24gb
#PBS -J 0-100:1
#PBS -N gplite_small_15
#PBS -A ex-kdd-1
#PBS -m abe    
#PBS -M rtkushner@alumni.ubc.ca                                      
#PBS -o ./outputs/output.txt
#PBS -e ./outputs/error.txt
 
################################################################################
 
module load python3

cd $PBS_O_WORKDIR

source ../polyvenv/bin/activate


python3 ../staticpopv6.py -sim $PBS_ARRAY_INDEX -t 5 -it 10000000 -SP 1 --outdir  $PBS_O_WORKDIR -ls 0.1

