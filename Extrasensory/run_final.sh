#!/bin/bash
results_file="results_final_alpha_05_t05_rerun.txt"
for test_context in "label:PHONE_IN_BAG" "label:PHONE_IN_HAND"
do
    for seed in '369284' '96431' '64140'
    do
        for user_id in '61976C24-1C50-4355-9C49-AAE44A7D09F6' '7CE37510-56D0-4120-A1CF-0E23351428D2' '806289BC-AD52-4CC1-806C-0CDB14D65EB6' '9DC38D04-E82E-4F29-AB52-B476535226F2' 'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C'
        do
            file_name="${user_id}_${test_context}_${seed}.out"
            nohup python -u run_EXTRASENSORY_PERSONAL_aaai.py ${test_context} ${seed} ${user_id} >> ${results_file}
        done
    done
done
