#!/bin/bash

#PBS -l walltime=3:00:00,select=1:ncpus=1:mem=4gb
#PBS -J 0-15:1
#PBS -N sim_lite_correlated
#PBS -A pr-kdd-1
#PBS -m abe    
#PBS -M rtkushner@alumni.ubc.ca                                      
#PBS -o ./outputs/output.txt
#PBS -e ./outputs/error.txt
 
################################################################################
 
module load python3

cd $PBS_O_WORKDIR

source ../polyvenv/bin/activate

python3 ../fall12.py -sim $PBS_ARRAY_INDEX -t 1 -itstart  8000001 -itend 9000000  --outdir $PBS_O_WORKDIR -ls 1 -SP 1 -re $PBS_O_WORKDIR

