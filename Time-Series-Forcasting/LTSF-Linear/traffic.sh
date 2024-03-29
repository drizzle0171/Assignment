
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/traffic-2021" ]; then
    mkdir ./logs/traffic-2021
fi

for model_name in Linear NLinear DLinear DLinear_T DLinear_S
do
seq_len=336
ver=TWM-ReLU

CUDA_VISIBLE_DEVICES=7 python -u run_longExp.py \
  --traffic 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 0 \
  --embed learned \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --time_channel 4 \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 >./logs/traffic-2021/$model_name'_'traffic'_'$ver'_'$seq_len'_'96'_'time_sigmoid.log 
printf "Done %s 96" ${model_name}

CUDA_VISIBLE_DEVICES=7 python -u run_longExp.py \
  --traffic 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 0 \
  --embed learned \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --time_channel 4 \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 >./logs/traffic-2021/$model_name'_'traffic'_'$ver'_'_$seq_len'_'192'_'time_sigmoid.log  
printf "Done %s 192" ${model_name}

CUDA_VISIBLE_DEVICES=7 python -u run_longExp.py \
  --traffic 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 0 \
  --embed learned \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --time_channel 4 \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 >./logs/traffic-2021/$model_name'_'traffic'_'$ver'_'_$seq_len'_'336'_'time_sigmoid.log  
printf "Done %s 336" ${model_name}

CUDA_VISIBLE_DEVICES=7 python -u run_longExp.py \
  --traffic 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 0 \
  --embed learned \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --time_channel 4 \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 >./logs/traffic-2021/$model_name'_'traffic'_'$ver'_'_$seq_len'_'720'_'time_sigmoid.log  
printf "Done %s 720" ${model_name}

done