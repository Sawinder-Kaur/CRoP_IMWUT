#!/bin/bash
for u_id in "0" "1" "2"
do
    for g_id in "0" "1" "2"
    do
        for training_set in "test" "train"
        do
            file_name="${u_id}_${training_set}_${g_id}.out"
            model_to_train_path="/home/sakaur/condor_files/dependencies/global_models/gid_${g_id}_uid_${u_id}.pt"
            nohup python -u run_WIDAR_PERSONAL_aaai_final.py "./extracted_data" "./split_data" './saved_models/' ${u_id} ${g_id} ${model_to_train_path} ${training_set} '1e-6' '600' '1' '0.9' 'MP_unstruct' "l1" "0.001" "0.10" &> ${file_name}
        done
    done
done
