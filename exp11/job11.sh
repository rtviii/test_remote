#!/bin/bash

#PBS -l walltime=48:00:00,select=1:ncpus=1:mem=8gb
#PBS -J 0-50:1
#PBS -N tens_web_large_corr
#PBS -A pr-kdd-1
#PBS -m abe    
#PBS -M rtkushner@alumni.ubc.ca                                      
#PBS -o ./outputs/output.txt
#PBS -e ./outputs/error.txt
 
################################################################################
 
module load python3

cd $PBS_O_WORKDIR
source ../venv/bin/activate
python3 ../tens_poisson.py -sim $PBS_ARRAY_INDEX -t 4 -itstart 0 -itend 10000000 -SP 1 --outdir $PBS_O_WORKDIR -ls 1

