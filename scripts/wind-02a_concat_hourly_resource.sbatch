#!/bin/bash -l

#SBATCH -J concat
#SBATCH -N 1					
#SBATCH -t 00:35:00
#SBATCH -o stdout/%x_%A_%a.out
#SBATCH -e err/%x_%A_%a.err
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
####SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --array=26,62,67,87,92,99,104

# Read the line corresponding to the task ID
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" input-list.dat)
# Parse into variables
read RANK DOMAIN MODEL YEAR <<< "$LINE"

# Start time tracking
start_time=$(date +%s)

log_file="stdout/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

# Print SBATCH job settings for debugging to the log file
{
  echo "====================================="
  echo "Network: $NETWORK"
  echo "Job Name: $SLURM_JOB_NAME"
  echo "Job ID: $SLURM_JOB_ID"
  echo "Partition: $SLURM_JOB_PARTITION"
  echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
  echo "Tasks Per Node: $SLURM_NTASKS_PER_NODE"
  echo "Total Tasks: $SLURM_NTASKS"
  echo "CPUs Per Task: $SLURM_CPUS_PER_TASK"
  echo "Job Start Time: $(date)"
  echo "====================================="
} >> "$log_file"

mkdir -p missing_times

# Now you can use $DOMAIN, $MODEL, $YEAR in your script
{
  echo "Running for:"
  echo "Rank: $IDX"
  echo "Domain: $DOMAIN"
  echo "Model: $MODEL"
  echo "Year: $YEAR"
} >> "$log_file"

# Load OpenMPI
module load openmpi

# Load Conda initialization 
source /shared/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate /shared/miniconda3/envs/renew

PYSCRIPT=../src/preprocess/wind_02a_concat_hourly_resource.py
python3 ${PYSCRIPT} --model ${MODEL} --domain ${DOMAIN} --year ${YEAR}

# End time tracking
end_time=$(date +%s)

# Calculate and print elapsed time to the log file
elapsed_time=$((end_time - start_time))
{
  echo "====================================="
  echo "Job completed in $elapsed_time seconds."
  echo "Job End Time: $(date)"
  echo "====================================="
} >> "$log_file"

# Deactivate Conda environment after job completion
conda deactivate


