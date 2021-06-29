#!/bin/bash

#PBS -l walltime=20:00:00,select=1:ncpus=1:mem=4gb
#PBS -J 0-100:1
#PBS -N popflat18
#PBS -A ex-kdd-1
#PBS -m abe    
#PBS -M rtkushner@alumni.ubc.ca                                      
#PBS -o ./outputs/output.txt
#PBS -e ./outputs/error.txt
 
################################################################################
 
module load python3

cd $PBS_O_WORKDIR

source ../polyvenv/bin/activate

python3 ../persist9.py -sim $PBS_ARRAY_INDEX -re $PBS_O_WORKDIR -t 8 -itstart 10000000 -itend 20000000 -SP -1 --outdir $PBS_O_WORKDIR -ls 0.1 -alm_rate 0.0001 -gpm_rate 0

