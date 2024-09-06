export CUDA_VISIBLE_DEVICES=0

model_name=Piformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96_piformer \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --period_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --d_model 128 \
  --c_out 8 \
  --base_model iTransformer \
  --batch_size 4\
  --des 'Exp' \
  --train_epochs 1\
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 