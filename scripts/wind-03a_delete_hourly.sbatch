#!/bin/bash

#SBATCH -J delete_hourly
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -o out/%x_%A.out
#SBATCH -e err/%x_%A.err
#SBATCH --partition=data-xfer
#SBATCH --ntasks=1

### DOMAIN=${1}
### SIM_ID=${2}

DOMAIN="d02"
SIM_ID="MIROC6"

base_dir="/data/wind_resources/hourly/${DOMAIN}/${SIM_ID}"
log_file="delete_hourly_${DOMAIN}_${SIM_ID}.log"

# List of years to exclude
# exclude_years=("2038" "2030" "1997")
exclude_years=("2150" "2250")

# Clear previous log
> "$log_file"

#for year in $(seq 1981 2010); do
for year in $(seq 1975 2099); do
    # Skip if year is in the exclude list
    if [[ " ${exclude_years[*]} " == *" $year "* ]]; then
        echo "Skipping $year" | tee -a "$log_file"
        continue
    fi

    folder="${base_dir}/${year}"
    
    if [[ -d "$folder" ]]; then
        echo "Deleting $folder" | tee -a "$log_file"
        rm -rf "$folder" 
        #find "$folder" -mindepth 1 -delete
    else
        echo "Not found: $folder" | tee -a "$log_file"
    fi
done

wait
exit

