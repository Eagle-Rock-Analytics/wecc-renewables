#!/bin/bash -l

#SBATCH -J pv_res                        # Job name.
#SBATCH --array=1-4                      # Job array size.                 
#SBATCH -N 1					         # Number of nodes.
#SBATCH -t 03:00:00                      # Job time limit in HH:MM:SS.
#SBATCH -o stdout/pv_res-%A-%a.out       # Standard output file: Specifies the path for the job's standard output. '%j' is replaced by the Job ID.
#SBATCH -e err/pv_res-%A-%a.err          # Standard error file: Specifies the path for the job's standard error. '%j' is replaced by the Job ID.
#SBATCH -p compute                       # Partition name: Submits the job to the 'compute' queue/partition.
#SBATCH -c 48                            # CPU count per task.   

# Define the path to your Python script
PYSCRIPT=../src/preprocess/pv_resource_to_h5.py 
# Read in list of parameters (one line per script run):
ARG_ID=$(cat input-list.dat | awk "NR==$SLURM_ARRAY_TASK_ID")
# Run python script with a command line argument
conda run -n renew python3 ${PYSCRIPT} ${ARG_ID}