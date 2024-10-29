#!/bin/bash
for seed in 10 144 12345
do
    for user_id in '0001' '0002' '0003'
    do
        file_name="personal_user_${user_id}_${seed}_test_f1.out"
        python -u main_personalize_general_test_f1.py ${user_id} ${seed} &> ${file_name}
    done
done
