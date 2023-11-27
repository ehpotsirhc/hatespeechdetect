#!/bin/bash

dpath=../../datasets/SBIC/perturbed/
suffix=csv
declare -a torun=(
    SBIC_robustness_DIV_swapTarget
    SBIC_robustness_INV_contract
    SBIC_robustness_INV_expand
    SBIC_robustness_INV_typo
    SBIC_robustness_MFT
    SBIC_robustness_MFT_exhaustive
    )


for fname in ${torun[@]}
do
    echo ========== running $fname... ==========
    python bert.py --dataset $dpath$fname.$suffix
    cp logs/bert_run.log logs/archived/bert_$fname.log
done


for fname in ${torun[@]}
do
    echo ========== running $fname... ==========
    python xclass.py --dataset $dpath$fname.$suffix
    cp logs/xclass_run.log logs/archived/xclass_$fname.log
done

