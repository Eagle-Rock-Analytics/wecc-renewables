#!/bin/bash

sim_id="$1"
module="$2"
folder="/data/${sim_id}/${module}_${sim_id}/logs/stdout"

for file in ${folder}/*.e*; do
  if grep -q "Found NaN values" "$file"; then
    grep "filename:" "$file" | awk -F'filename: ' '{print $2}'
  fi
done | sort -u

#for file in ${folder}/*.e*; do
#  if grep -q "Found NaN values" "$file"; then
#    echo $file
#  fi
#done 
