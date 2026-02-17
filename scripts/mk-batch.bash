#!/bin/bash -l

# Makes list of arguments for pre + post-processing resource data (annual profiles only)
# Saves arguments for each run as a line in input-list.dat (preprocessing) and output-list.dat (postprocessing)
ONAME=output-list.dat
INAME=input-list.dat

rm -f $ONAME
rm -f $INAME

declare -a SIMS=("EC-Earth3" "MPI-ESM1-2-HR" "MIROC6" "TaiESM1" "ERA5")
# declare -a SIMS=("MIROC6")

declare -a MODS=("utility" "distributed" "offshore" "onshore")

declare -a TECH=("PVWattsv8" "Windpower")
# declare -a TECH=("Windpower")

#declare -a MODS=("offshore")

DOMAIN=d02
START=1981
END=2013

i=$(( $START - 1980 ))

for j in "${SIMS[@]}"; do
    for year in $(seq $START $END); do
        echo "${i} ${DOMAIN} ${j} ${year}" >> $INAME
	((i++))
        for k in "${MODS[@]}"; do
	    o=$(( $i - 1 ))	
            echo "${o} ${DOMAIN} ${j} ${k} ${TECH} ${year}" >> $ONAME
	    ((counter++))
        done
    done
done

