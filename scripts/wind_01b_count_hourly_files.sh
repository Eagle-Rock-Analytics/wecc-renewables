#!/bin/bash -l
#SBATCH --job-name=wind_d02         # Name of the job
#SBATCH --array=1-118               # Number of lines in networks-input.dat  
#SBATCH --ntasks=108                # Total number of MPI processes (1 per CPU)
#SBATCH --nodes=6                   # Number of nodes (1 node with 36 CPUs)
#SBATCH --ntasks-per-node=18        # Number of MPI processes per node (36 processes per node)
#SBATCH --cpus-per-task=2           # Number of MPI processes per node (36 processes per node)
#SBATCH --time=01:00:00             # Maximum runtime (adjust as necessary)
#SBATCH --partition=small-compute   # Queue/partition (small-compute has 36 vcores)
#SBATCH --output=out/%x_%A_%a.out   # Standard output file with job name and job ID
#SBATCH --error=err/%x_%A_%a.err    # Standard error file with job name and job ID

# Load OpenMPI
module load openmpi

log_file="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

# Parse model, year and domain from input-list.dat
# Read the line corresponding to the task ID
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" input-list.dat)

# Parse into variables
read IDX DOMAIN MODEL YEAR <<< "$LINE"

# Now you can use $DOMAIN, $MODEL, $YEAR in your script
{
  echo "Running for:"
  echo "Index: $IDX"
  echo "Domain: $DOMAIN"
  echo "Model: $MODEL"
  echo "Year: $YEAR"
} >> "$log_file"

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

# Define the path to your Python script
PYSCRIPT="../src/preprocess/wind_01a_hourly_resource_from_json.py"

# Check if the Python script exists
if [ ! -f "$PYSCRIPT" ]; then
  echo "Error: Python script not found!" >> "$log_file"
  exit 1
fi

# Load Conda initialization 
source /shared/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate /shared/miniconda3/envs/renew

# Start time tracking
start_time=$(date +%s)

# Run the Python script directly using Python from the activated environment
srun --mpi=pmi2 python3 ${PYSCRIPT} --model $MODEL --domain $DOMAIN --year $YEAR

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
