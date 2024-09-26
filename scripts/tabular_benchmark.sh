#!/bin/bash

SEED=$1
NFOLDS=5
NTRIALS=2
VALSIZE=.2

# declare -a datasets=(
#     "compas-two-years clf_cat"
#     "california clf_num"
#     "electricity clf_cat"
#     "albert clf_cat"
#     "bank-marketing clf_num"
#     "MagicTelescope clf_num"
#     "house_16H clf_num"
#     "heloc clf_num"
#     "default-of-credit-card-clients clf_cat"
#     "default-of-credit-card-clients clf_num"
# )
declare -a datasets=(
    "compas-two-years clf_cat"
    "bank-marketing clf_num"
)


for elem in "${datasets[@]}"
do
	read -a strarr <<< "$elem"  # uses default whitespace IFS
	fname="${strarr[1]}_${strarr[0]}_${SEED}"
	out_dir="./slurm_logs/${fname}_out"
	err_dir="./slurm_logs/${fname}_err"
	echo "${fname}"
	sleep 4
	sbatch -J $fname -e $err_dir -o $out_dir -t 0-15:00:00 --mem 16G ./scripts/slurm.job ${strarr[0]} ${strarr[1]} ${NFOLDS} ${NTRIALS} ${SEED} ${VALSIZE}
	sleep 4
done
