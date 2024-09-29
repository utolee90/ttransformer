model_name=iTransformer
model_name2=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_96\
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
  --train_ratio 0.7 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate_25mean.csv \
  --model_id iTransformer_Exchange_96_96_25mean\
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
  --train_ratio 0.7 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate_101mean.csv \
  --model_id iTransformer_Exchange_96_96_101mean\
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
  --train_ratio 0.7 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 
  
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate_401mean.csv \
  --model_id iTransformer_Exchange_96_96_401mean\
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
  --train_ratio 0.7 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_192\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 8 \
  --train_ratio 0.7 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate_25mean.csv \
  --model_id iTransformer_Exchange_96_192_25mean\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 8 \
  --train_ratio 0.7 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate_101mean.csv \
  --model_id iTransformer_Exchange_96_192_101mean\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 8 \
  --train_ratio 0.7 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 
  
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate_401mean.csv \
  --model_id iTransformer_Exchange_96_96_401mean\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 8 \
  --train_ratio 0.7 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 
  
