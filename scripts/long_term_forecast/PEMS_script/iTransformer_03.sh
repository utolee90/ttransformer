export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

seq_len=96

for pred_len in 12 24 48 96
do 
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 4 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --learning_rate 0.001 \
    --itr 1\
    --train_ratio 0.6

done