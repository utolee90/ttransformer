export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=DiTransformer
seq_len=96

for pred_len in 96 192 336 720;
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --d_model 512\
    --d_ff 512\
    --itr 1 > logs/LongForecasting/$model_name'_'tester_$seq_len'_'$pred_len.log 
done

