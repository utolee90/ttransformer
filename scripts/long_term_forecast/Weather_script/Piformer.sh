export CUDA_VISIBLE_DEVICES=0

model_name=Piformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --base_model DLinear \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1 \
  --period_len 24 \
  --train_epochs 1