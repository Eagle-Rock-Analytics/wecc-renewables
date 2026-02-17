#!/bin/bash

#SBATCH -J day-agg                     # Job name.
#SBATCH --array=1-36                   # Job array size.
#SBATCH -o stdout/%J-%a.out            # Path for the job's standard output. '%j' is replaced by the Job ID.
#SBATCH -e err/%J-%a.err               # Path for the job's standard error. '%j' is replaced by the Job ID.
#SBATCH -p compute                     # Partition name: Submits the job to the 'compute' queue/partition.
#SBATCH -c 32                          # CPUs per task.
#SBATCH -t 02:00:00                    # Job time limit in HH:MM:SS.

# Define the path to your Python script
PYSCRIPT=../src/postprocess/all_hourly_to_daily.py 
# Read in list of parameters (one line per script run):
DAT_ID=$(cat agg-list.dat | awk "NR==$SLURM_ARRAY_TASK_ID")
# Run python script with a command line argument
conda run -n renew python3 ${PYSCRIPT} ${DAT_ID}