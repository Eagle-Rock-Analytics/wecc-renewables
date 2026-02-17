#!/bin/bash

#SBATCH -J jsons
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o out/%x_%A.out
#SBATCH -e err/%x_%A.err
#SBATCH --partition=small-compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Define the path to your Python script
PYSCRIPT="../src/preprocess/wind_00_create_json.py"

# Load Conda initialization
source /shared/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate /shared/miniconda3/envs/renew

# Start time tracking
start_time=$(date +%s)

# Run the Python script directly using Python from the activated environment
python3 ${PYSCRIPT} --domain d02 &
python3 ${PYSCRIPT} --domain d03 &
wait

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
wait
exit

