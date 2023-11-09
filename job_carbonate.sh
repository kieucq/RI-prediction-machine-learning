#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J otis_RI
#SBATCH -p gpu --gpus-per-node v100:1
#SBATCH -A r00043 
#SBATCH --mem=128G
#module load PrgEnv-gnu
module load python/gpu
cd /N/u/ckieu/BigRed200/model/RI-prediction
#
# setting up the ML experiments
#
nloop=10
stormname="OTIS"
exp_name="metric_sensitivity_0p7"
#
# loop over all realizations
#
rm -f out_${exp_name}.txt log*.txt
loop=1
while [ ${loop} -le ${nloop} ]; do
    echo "Running realization $loop ..."
    python RI_deep_learning_otis.py > log${loop}.txt
    echo "Realization # ${loop}" >> out_${exp_name}.txt
    grep F1 log${loop}.txt >> out_${exp_name}.txt
    loop=$(( $loop + 1 ))
done
