#!/bin/bash
#SBATCH -A csc538
#SBATCH -J Weather
#SBATCH -o ../../job_logs/ts-transfer/%x-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1

export PATH="/ccs/home/rolandriachi/ts-transfer/ofa-env/bin:$PATH"
export SCRATCH="/lustre/orion/csc538/scratch/rolandriachi"
export PROJ_SHARE="/lustre/orion/csc538/proj-shared/llm2ts/"
module load cray-python/3.9 rocm/5.6.0
source ~/ts-transfer/ofa-env/bin/activate

model_name=GPT4TS
root_path=$SCRATCH

# export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=GPT4TS

for percent in 100
do
for pred_len in 96 # 192 336 720
do

python main.py \
    --root_path $PROJ_SHARE/weather/ \
    --data_path weather.csv \
    --model_id weather_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 512 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --lradj type3 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --is_gpt 1
    
done
done