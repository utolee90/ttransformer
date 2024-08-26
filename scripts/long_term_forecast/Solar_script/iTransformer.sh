export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

seq_len=96
for pred_len in 96 192
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id solar_$seq_len'_'$pred_len \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --learning_rate 0.0005 \
    --itr 1
done
