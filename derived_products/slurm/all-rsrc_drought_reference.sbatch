#!/bin/bash

#SBATCH -J rd_ref
#SBATCH --array=1-4
#SBATCH -o stdout/%J-%a.out
#SBATCH -e err/%J-%a.err
#SBATCH -p compute
#SBATCH -c 32

PYSCRIPT=/shared/renewable-profiles/scripts/derived_products/all-rsrc_drought_reference.py
DAT_ID=$(cat gwl-list.dat | awk "NR==$SLURM_ARRAY_TASK_ID")
# Run python script with a command line argument
conda run -n renew python3 ${PYSCRIPT} ${DAT_ID}
