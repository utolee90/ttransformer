export CUDA_VISIBLE_DEVICES=0

model_name=PITS_decomp

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_48_piformer \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 24 \
  --pred_len 96 \
  --period_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --d_model 512 \
  --c_out 8 \
  --batch_size 8\
  --des 'Exp' \
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 
