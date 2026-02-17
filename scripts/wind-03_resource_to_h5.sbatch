#!/bin/bash

#SBATCH -J wind_res
#SBATCH --array=36,51                 
#SBATCH -N 1					
#SBATCH -t 03:00:00
#SBATCH -o stdout/wind_res-%A-%a.out
#SBATCH -e err/wind_res-%A-%a.err
#SBATCH -c 24
#SBATCH --partition=compute
#SBATCH --exclusive

PYSCRIPT=../src/preprocess/wind_03b_resource_to_h5.py 
ARG_ID=$(cat input-list.dat | awk "NR==$SLURM_ARRAY_TASK_ID")
echo $ARG_ID
conda run -n renew python3 ${PYSCRIPT} ${ARG_ID}
