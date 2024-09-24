export CUDA_VISIBLE_DEVICES=0

model_name=DiTransformer_decomp

models=("DiTransformer_decomp" "DLinear_trend")
for model_name in "${models[@]}"
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id DiTransformer_Exchange_96_96_lin2_origin8\
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
  --batch_size 4 \
  --train_ratio 0.6 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --shuffle 0\
  --itr 1 # > logs/LongForecasting/Exchange_RATE_$model_name'_'tester_$seq_len'_'$pred_len.log 
done
