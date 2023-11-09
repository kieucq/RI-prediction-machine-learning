#!/bin/bash
module load python/gpu
storm_name="PATRICIA20E"
storm_cycles=13
storm_start="2015102106"
python ./RI_getSHIP.py cycles ${storm_cycles} ${storm_name} ${storm_start}
head -n 1 ${storm_name}.${storm_start}.ship.txt > ${storm_name}_master.csv
cat ${storm_name}*ship.txt | sed '/Storm/d'  >> ${storm_name}_master.csv
