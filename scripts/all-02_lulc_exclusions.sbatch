#!/bin/bash

#SBATCH -J excl-meta                   # Job name.
#SBATCH -N 1                           # Node count.
#SBATCH --array=1-4                    # Job array size.
#SBATCH -c 96                          # CPU count per task.
#SBATCH -t 36:00:00                    # Time limit in HH:MM:SS.
#SBATCH -o stdout/excl-meta-%j.out     # Path for the job's standard output. '%j' is replaced by the Job ID.
#SBATCH -e err/excl-meta-%j.err        # Path for the job's standard error. '%j' is replaced by the Job ID.
#SBATCH -p compute                     # Partition name: Submits the job to the 'compute' queue/partition.

# Define the path to your Python script
PYSCRIPT=../src/preprocess/all_02_lulc_exclusions.py
# Set the domain:
DOMAIN=d02
# Read in list of exclusions being added to the spatial HDF5:
EXCLUSION=$(cat list-exclusions.dat | awk "NR==$SLURM_ARRAY_TASK_ID")
# Run it
conda run -n renew python3 ${PYSCRIPT} ${DOMAIN} ${EXCLUSION}
