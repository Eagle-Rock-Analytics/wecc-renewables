#!/bin/bash

domain="${1}"
sim_id="${2}"

output="out/zarr_files_size_${sim_id}_${domain}.dat"
> $output

for year in {1981..2098}; do 

  size=$(aws s3api list-objects-v2   --bucket wfclimres   --prefix era/resource_data/${domain}/wind/${sim_id}/zarrs/${sim_id}_${domain}_wind_resource_${year}.zarr/   --query "sum(Contents[].Size)"   --output text)
  total=0; for num in $size; do total=$((total + num)); done
  total=$(echo "scale=2; $total / 1000000000" | bc)
  
  # total="`aws s3 ls s3://wfclimres/era/resource_data/${domain}/wind/${sim_id}/zarrs/${sim_id}_${domain}_wind_resource_${year}.zarr/ --recursive --summarize --human-readable --output text | egrep "Total Size";`" 
  
  echo "${year} : Total size: $total GB"
  echo "${year} : Total size: $total GB" >> $output

done
