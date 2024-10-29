#!/bin/bash
for seed in 1013 #144 12345
do
    for user_id in '17' '25' '28' '336' '344' '361' '362' '55' '586' '587' '589' '590' '591' '61' '67' '80'
    do
        file_name="test_seeds.out"
        python -u main_test.py ${user_id} ${seed} 'overall' >> ${file_name}
    done
done
