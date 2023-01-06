#!/bin/bash

submits_dir="${HOME}/workspace/DeepGlobe-Road-Extraction-Challenge/submits/"

for fold in 0 1 2 3 4 5; do

  echo -e "\n##############\n### FOLD ${fold} ###\n##############\n"

  data_dir="${HOME}/data/northern-cities/gillam_mb_canada/fold-000${fold}/test/"
  output_dirs="${submits_dir}/dlinknet34-imagenet-gillam-all-season-fold-000${fold}-*/"

  for output_dir in ${output_dirs}; do
    python -u utils/dataset_coregister.py \
           --data_dir "${data_dir}" \
           --output_dir "${output_dir}"
  done

done
