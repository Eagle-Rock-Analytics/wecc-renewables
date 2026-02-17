#!/bin/bash -l
# Makes list of arguments for resource droughts under given global warming levels
ONAME=gwl-list.dat
rm -f $ONAME
declare -a SIMS=("ERA5" "EC-Earth3" "MPI-ESM1-2-HR" "MIROC6" "TaiESM1")
#declare -a SIMS=("ERA5")
declare -a TECHS=("pv" "windpower")
declare -a VARS=("gen")
# choices for global warming level:
declare -a GWLS=("0.8" "1.0" "1.2" "1.5" "2.0" "2.5" "3.0" "4.0")
# use 0.8 for calculating the reference production for resource drought identification
DOMAIN=d03

for SIM in "${SIMS[@]}"; do
	if [ ${SIM} = "ERA5" ]; then
		declare -a GWLS=("nan")
	else
		declare -a GWLS=("0.8" "1.0" "1.2" "1.5" "2.0" "2.5" "3.0" "4.0")
	fi
	for TECH in "${TECHS[@]}"; do
		if [ ${TECH} = "pv" ]; then
			declare -a MODS=("utility" "distributed")
		elif [ ${TECH} = "windpower" ]; then
			declare -a MODS=("offshore" "onshore")
		fi
		for MOD in "${MODS[@]}"; do
			for GWL in "${GWLS[@]}"; do
				for VAR in "${VARS[@]}"; do
					echo "${DOMAIN} ${SIM} ${MOD} ${GWL} ${VAR} ${TECH}" >> $ONAME
				done
			done
		done
	done
done
