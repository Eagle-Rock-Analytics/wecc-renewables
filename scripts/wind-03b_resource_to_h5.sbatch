#!/bin/bash -l

#SBATCH -J wind2h5
#SBATCH -N 1					
#SBATCH -t 01:00:00
#SBATCH -o stdout/wind2h5-%A-%a.out
#SBATCH -e err/wind2h5-%A-%a.err
#SBATCH --array=1-33    # t0
#####SBATCH --array=35-64    #t1
#####SBATCH --array=65-94    #t2
#####SBATCH --array=95-118   #t3
#SBATCH -p compute
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G

PYSCRIPT=../src/preprocess/wind_03b_resource_to_h5.py  
#ARG_ID=$(cat input-list.dat | awk "NR==$SLURM_ARRAY_TASK_ID")
ARG_ID=$(awk -v id="$SLURM_ARRAY_TASK_ID" '$1 == id {print $0}' input-list.dat)

conda run -n renew python3 ${PYSCRIPT} ${ARG_ID}

