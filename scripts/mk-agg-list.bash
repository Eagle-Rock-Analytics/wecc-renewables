#!/bin/bash
# Makes list of arguments for aggregating hourly profiles to daily
ONAME=agg-list.dat
rm -f $ONAME
declare -a SIMS=("EC-Earth3" "MPI-ESM1-2-HR" "MIROC6" "TaiESM1" "ERA5")
declare -a TECHS=("windpower" "pv")
declare -a VARS=("cf" "gen")
DOMAIN=d03

for SIM in "${SIMS[@]}"; do
	if [ ${SIM} = "ERA5" ]; then
		declare -a SCENS=("reanalysis")
	else
		declare -a SCENS=("historical" "ssp370")
	fi
	for TECH in "${TECHS[@]}"; do
		if [ ${TECH} = "pv" ]; then
			declare -a MODS=("utility" "distributed")
		else
			declare -a MODS=("offshore" "onshore")
		fi
		for MOD in "${MODS[@]}"; do
			for SCEN in "${SCENS[@]}"; do
				for VAR in "${VARS[@]}"; do
					echo "${DOMAIN} ${SIM} ${MOD} ${SCEN} ${VAR} ${TECH}" >> $ONAME
				done
			done
		done
	done
done
