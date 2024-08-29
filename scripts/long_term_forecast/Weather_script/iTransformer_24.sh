# 24-24 model tester

python -u run.py --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_24_24_ts \
  --model iTransformer \
  --data custom \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 64\
  --d_ff 64\
  --itr 1