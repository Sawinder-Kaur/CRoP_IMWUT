#!/bin/bash
for seed in 2345 #1013 #144 #
do
    for user_id in '17' '25' '28' '336' '344' '361' '362' '55' '586' '587' '589' '590' '591' '61' '67' '80'
    do
        file_name="personalize_user_${user_id}_${seed}_bilstm.out"
        python -u main_personal_finetune_trainable_alpha.py ${user_id} ${seed} &> ${file_name}
    done
done
