#!/bin/bash

#SBATCH -J meta                   # Job name.
#SBATCH -N 1                      # Node count.
#SBATCH -c 96                     # CPU count per task.
#SBATCH -t 08:00:00               # Time limit in HH:MM:SS. 
#SBATCH -o stdout/meta-%j.out     # Standard output file: Specifies the path for the job's standard output. '%j' is replaced by the Job ID.
#SBATCH -e err/meta-%j.err        # Standard error file: Specifies the path for the job's standard error. '%j' is replaced by the Job ID.
#SBATCH -p compute                # Partition name: Submits the job to the 'compute' queue/partition.

# Define the path to your Python script
PYSCRIPT=../src/preprocess/all_01_meta_node_create.py
# Specify parameters
DOMAIN=d02
# Run it
conda run -n renew python3 -u ${PYSCRIPT} ${DOMAIN}
