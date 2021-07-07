#!/bin/bash

#PBS -l walltime=30:00:00,select=1:ncpus=1:mem=8gb
#PBS -J 0-100:1
#PBS -N fall
#PBS -A pr-kdd-1
#PBS -m abe    
#PBS -M rtkushner@alumni.ubc.ca                                      
#PBS -o ./outputs/output.txt
#PBS -e ./outputs/error.txt
 
################################################################################
 
module load python3

cd $PBS_O_WORKDIR

source ../polyvenv/bin/activate

python3 ../fall12.py -sim $PBS_ARRAY_INDEX  -itstart 0 -itend 25000000 --outdir $PBS_O_WORKDIR -ls 1 -t 1 -SP 1  

