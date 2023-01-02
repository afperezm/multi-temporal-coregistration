#!/bin/bash

echo -e "\n##############\n### FOLD 0 ###\n##############\n"

data_dir="${HOME}/data/northern-cities/gillam_mb_canada/fold-0000/test/"

for ckpt_dir in weights/dlinknet34-imagenet-gillam-all-season-fold-0000-*; do
  python -u test.py \
         --data_dir "${data_dir}" \
         --output_dir ./submits/ \
         --checkpoints_dir "${ckpt_dir}" \
         --model "model"
done

echo -e "\n##############\n### FOLD 1 ###\n##############\n"

data_dir="${HOME}/data/northern-cities/gillam_mb_canada/fold-0001/test/"

for ckpt_dir in weights/dlinknet34-imagenet-gillam-all-season-fold-0001-*; do
  python -u test.py \
         --data_dir "${data_dir}" \
         --output_dir ./submits/ \
         --checkpoints_dir "${ckpt_dir}" \
         --model "model"
done

echo -e "\n##############\n### FOLD 2 ###\n##############\n"

data_dir="${HOME}/data/northern-cities/gillam_mb_canada/fold-0002/test/"

for ckpt_dir in weights/dlinknet34-imagenet-gillam-all-season-fold-0002-*; do
  python -u test.py \
         --data_dir "${data_dir}" \
         --output_dir ./submits/ \
         --checkpoints_dir "${ckpt_dir}" \
         --model "model"
done

echo -e "\n##############\n### FOLD 3 ###\n##############\n"

data_dir="${HOME}/data/northern-cities/gillam_mb_canada/fold-0003/test/"

for ckpt_dir in weights/dlinknet34-imagenet-gillam-all-season-fold-0003-*; do
  python -u test.py \
         --data_dir "${data_dir}" \
         --output_dir ./submits/ \
         --checkpoints_dir "${ckpt_dir}" \
         --model "model"
done

echo -e "\n##############\n### FOLD 4 ###\n##############\n"

data_dir="${HOME}/data/northern-cities/gillam_mb_canada/fold-0004/test/"

for ckpt_dir in weights/dlinknet34-imagenet-gillam-all-season-fold-0004-*; do
  python -u test.py \
         --data_dir "${data_dir}" \
         --output_dir ./submits/ \
         --checkpoints_dir "${ckpt_dir}" \
         --model "model"
done

echo -e "\n##############\n### FOLD 5 ###\n##############\n"

data_dir="${HOME}/data/northern-cities/gillam_mb_canada/fold-0005/test/"

for ckpt_dir in weights/dlinknet34-imagenet-gillam-all-season-fold-0005-*; do
  python -u test.py \
         --data_dir "${data_dir}" \
         --output_dir ./submits/ \
         --checkpoints_dir "${ckpt_dir}" \
         --model "model"
done
