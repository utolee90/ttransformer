export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
model_name2=iTransformer_linear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id COMPARE_iTransformer_Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 8 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --is_training 0 \
  --checkpoint_name "long_term_forecast_COMPARE_iTransformer_Exchange_96_96_Mod-iTransformer_data-exchange_rate.csv_(96to96)_0(1729846858)" \
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id COMPARE_iTransformer_linear_Exchange_96_96 \
#   --model $model_name2 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --batch_size 8 \
#   --d_model 64 \
#   --d_ff 128 \
#   --des 'Exp' \
#   --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 