#!/bin/bash

# Makes list of arguments for post-processing annual ReV output zarrs and packaging into scenario-wide profiles
# Saves arguments for each run as a line in package-list.dat
ONAME=package-list.dat
rm -f $ONAME
declare -a MODS=("utility" "distributed" "offshore" "onshore")
declare -a SIMS=("EC-Earth3" "MPI-ESM1-2-HR" "MIROC6" "TaiESM1" "ERA5")
# declare -a MODS=("onshore" "offshore")
DOMAIN=d03

for i in "${SIMS[@]}"; do
	if [ ${i} = "ERA5" ]; then
		declare -a SCENS=("reanalysis")
	else
		declare -a SCENS=("historical")
	fi
	for j in "${MODS[@]}"; do
		for k in "${SCENS[@]}"; do
			echo "${DOMAIN} ${i} ${j} ${k}" >> $ONAME
		done
	done
done
