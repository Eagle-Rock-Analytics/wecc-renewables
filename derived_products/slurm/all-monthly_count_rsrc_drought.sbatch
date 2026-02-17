#!/bin/bash

#SBATCH -J rd_detect
#SBATCH --array=1-132
#SBATCH -o stdout/rd_detect-%J-%a.out
#SBATCH -e err/rd_detect-%J-%a.err
#SBATCH -p compute
#SBATCH -c 4

PYSCRIPT=/shared/renewable-profiles/scripts/derived_products/all-monthly_count_rsrc_drought.py
DAT_ID=$(cat gwl-list.dat | awk "NR==$SLURM_ARRAY_TASK_ID")
# Run python script with a command line argument
conda run -n renew python3 ${PYSCRIPT} ${DAT_ID}
