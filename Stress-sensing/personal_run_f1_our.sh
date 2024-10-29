#!/bin/bash
for seed in 10 144 12345
do
    for user_id in '0001' '0002' '0003'
    do
        file_name="personalize_user_${user_id}_${seed}_dnn.out"
        python -u main_personalize_general_f1_our_trainable_alpha.py ${user_id} ${seed} &> ${file_name}
    done
done
