# iTransformer용 최적화된 스크립트

model_name=iTransformer
base_idea=Weather
root_path=./dataset/weather/
data_path=weather.csv

#START_SCRIPT_1
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
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
  --checkpoint_name "long_term_forecast_COMPARE_"${model_name}"_"${base_idea}"_96_96_Mod-"${model_name}"_data-"${data_path}"_(96to96)_ENSEMBLE" \
  --itr 1
# 
#START_SCRIPT_2
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ${root_path} \
  --data_path ${data_path} \
  --model_id COMPARE_${model_name}_${base_idea}_96_192 \
  --model ${model_name} \
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
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --checkpoint_name "long_term_forecast_COMPARE_"${model_name}"_"${base_idea}"_96_192_Mod-"${model_name}"_data-"${data_path}"_(96to192)_ENSEMBLE" \
  --itr 1
#
#START_SCRIPT_3
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ${root_path} \
  --data_path ${data_path} \
  --model_id COMPARE_${model_name}_${base_idea}_96_336 \
  --model ${model_name} \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
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
  --checkpoint_name "long_term_forecast_COMPARE_"${model_name}"_"${base_idea}"_96_336_Mod-"${model_name}"_data-"${data_path}"_(96to336)_ENSEMBLE" \
  --itr 1
#
#START_SCRIPT_4
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ${root_path} \
  --data_path ${data_path} \
  --model_id COMPARE_${model_name}_${base_idea}_96_720 \
  --model ${model_name} \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
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
  --checkpoint_name "long_term_forecast_COMPARE_"${model_name}"_"${base_idea}"_96_720_Mod-"${model_name}"_data-"${data_path}"_(96to720)_ENSEMBLE" \
  --itr 1
#
