#!/bin/bash

data_dir="${HOME}/data/northern-cities/gillam_mb_canada/"

python -u utils/baseline_coregister.py \
       --data_dir "${data_dir}/folds/S2_10m/" \
       --output_dir "${data_dir}/folds/S2_10m_deshifted/" \
       --match_band 3

for fold in 0 1 2 3 4 5; do

  echo -e "\n##############\n### FOLD ${fold} ###\n##############\n"

  ln -s "${data_dir}/folds/fold-000${fold}/train_wkt.csv" "${data_dir}/folds/fold-000${fold}/train_deshifted_wkt.csv"
  ln -s "${data_dir}/folds/fold-000${fold}/test_wkt.csv" "${data_dir}/folds/fold-000${fold}/test_deshifted_wkt.csv"

  for phase in train_deshifted test_deshifted; do

    python -u utils/dataset_images.py \
           --data_dir "${data_dir}/folds/fold-000${fold}/" \
           --images_dir "${data_dir}/folds/S2_10m_deshifted/" \
           --phase "${phase}" \
           --threshold 0.001 \
           --num_workers 8

  done

  rm "${data_dir}/folds/fold-000${fold}/train_deshifted_wkt.csv"
  rm "${data_dir}/folds/fold-000${fold}/test_deshifted_wkt.csv"

done
