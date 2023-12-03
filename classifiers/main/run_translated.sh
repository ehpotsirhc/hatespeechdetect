#!/bin/bash

dpath=../../datasets/SBIC/translated/
suffix=csv
declare -a torun=(
    translated_es2
    translated_hi2
    translated_ja2
    translated_ru2
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
