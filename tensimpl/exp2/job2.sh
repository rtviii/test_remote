#!/bin/bash

#PBS -l walltime=15:00:00,select=1:ncpus=1:mem=4gb
#PBS -J 0-100:1
#PBS -N tensor_impl2
#PBS -A pr-kdd-1
#PBS -m abe    
#PBS -M rtkushner@alumni.ubc.ca                                      
#PBS -o ./outputs/output.txt
#PBS -e ./outputs/error.txt
 
################################################################################
 
module load python3

cd $PBS_O_WORKDIR

source ../venv/bin/activate

python3 ../tensor_impl.py -sim $PBS_ARRAY_INDEX -t 1 -itstart 0 -itend 5000000 -SP 0 --outdir $PBS_O_WORKDIR -ls 0.3

