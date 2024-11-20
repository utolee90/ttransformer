# iTransformer용 최적화된 스크립트

model_name=TD_LTSF_Time
base_idea=Exchange
root_path=./dataset/exchange_rate/
data_path=exchange_rate.csv

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
  --batch_size 2 \
  --d_model 64 \
  --d_ff 128 \
  --shuffle 0 \
  --des 'Exp' \
  --individual 1 \
  --train_epochs 20 \
  --checkpoint_name "EXP_long_term_forecast_noshuffle_"${model_name}"_"${base_idea}"_96_96_Mod-"${model_name}"_data-"${data_path}"_(96to96)_ENSEMBLE_batch2" \
  --itr 1
# 
