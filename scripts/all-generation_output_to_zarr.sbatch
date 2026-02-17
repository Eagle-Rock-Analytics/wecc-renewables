#!/bin/bash

#SBATCH -J zarrit                    # Job name.
#SBATCH --array=1-33                 # Job array size.
#SBATCH -t 00:15:00                  # Time limit in HH:MM:SS
#SBATCH -o stdout/zarr-%A-%a.out     # Path for the job's standard output. '%j' is replaced by the Job ID.
#SBATCH -e err/zarr-%A-%a.err        # Path for the job's standard error. '%j' is replaced by the Job ID.
#SBATCH -p compute                   # Partition name: Submits the job to the 'compute' queue/partition.
#SBATCH -c 2                         # Number of CPUs per task.

# Define the path to your Python script
PYSCRIPT=../src/postprocess/all_generation_output_to_zarr.py 
# Read in list of parameters (one line per script run):
DAT_ID=$(awk -v id="$SLURM_ARRAY_TASK_ID" '$1 == id {print $0}' output-list.dat)
# Run python script with a command line argument
conda run -n renew python3 ${PYSCRIPT} ${DAT_ID}
