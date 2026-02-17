#!/bin/bash

#SBATCH -J zarrit                         # Job name.
#SBATCH --array=1                         # Job array size.
#SBATCH -o stdout/scenario-%A-%a.out      # Standard output file: Specifies the path for the job's standard output. '%j' is replaced by the Job ID.
#SBATCH -e err/scenario-%A-%a.err         # Standard error file: Specifies the path for the job's standard error. '%j' is replaced by the Job ID.
#SBATCH -p compute                        # Partition name: Submits the job to the 'compute' queue/partition.
#SBATCH -c 48                             # CPU count per task.      

# Define the path to your Python script
PYSCRIPT=../src/postprocess/wind_profiles_to_catalog.py
# Read in list of parameters (one line per script run):
DAT_ID=$(cat package-list.dat | awk "NR==$SLURM_ARRAY_TASK_ID")
# Run python script with a command line argument
conda run -n renew python3 ${PYSCRIPT} ${DAT_ID}
