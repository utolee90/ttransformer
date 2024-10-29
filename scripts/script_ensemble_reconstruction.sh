# iTransformer용 최적화된 스크립트

model_name=iTransformer
base_idea=Exchange
root_path=./dataset/exchange_rate/
data_path=exchange_rate.csv

#START_SCRIPT_1
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ${root_path} \
  --data_path ${data_path} \
  --model_id COMPARE_${model_name}_${base_idea}_96_96 \
  --model ${model_name} \
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
  --checkpoint_name "long_term_forecast_COMPARE_iTransformer_linear_Exchange_96_96_Mod-iTransformer_linear_data-exchange_rate.csv_(96to96)_0(1729900000)" \
  --itr 1
