export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

seq_len=96
for pred_len in 96
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/toy/ \
    --data_path toy.npz \
    --model_id toy_$seq_len'_'$pred_len \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 4 \
    --enc_in 5 \
    --dec_in 5 \
    --c_out 5 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --learning_rate 0.0005 \
    --itr 1
done
