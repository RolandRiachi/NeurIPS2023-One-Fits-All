#!/bin/bash
#SBATCH -A csc538
#SBATCH -J M4Yearly
#SBATCH -o ../../job_logs/ts-transfer/%x-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1

export PATH="/ccs/home/rolandriachi/ts-transfer/ofa-env/bin:$PATH"
export SCRATCH="/lustre/orion/csc538/scratch/rolandriachi/"
module load cray-python/3.9 rocm/5.4.0

model_name=GPT4TS
root_path=$SCRATCH/m4

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --gpt_layer 6 \
  --d_ff 128 \
  --d_model 128 \
  --patch_size 1 \
  --stride 1 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.002 \
  --loss 'SMAPE'