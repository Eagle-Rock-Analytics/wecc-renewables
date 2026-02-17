#!/bin/bash -l

#SBATCH -J edits
#SBATCH --array=1-9                   					
#SBATCH -o stdout/edit-%A-%a.out
#SBATCH -e err/edit-%A-%a.err
#SBATCH --ntasks=1

PYSCRIPT=../src/preprocess/misc_edit_pv_configs.py 
ARG_ID=$(cat input-list.dat | awk "NR==$SLURM_ARRAY_TASK_ID")
conda run -n renew python3 ${PYSCRIPT} ${ARG_ID}
