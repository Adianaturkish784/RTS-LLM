model_name=RTSLLM
train_epochs=10
learning_rate=1e-4

master_port=00097
batch_size=32
d_model=16
d_ff=32

llm_model=BERT
if [ "$llm_model" == "BERT" ]; then
  llm_layers=12
  llm_dim=768
elif [ "$llm_model" == "GPT2" ]; then
  llm_layers=12
  llm_dim=768
elif [ "$llm_model" == "LLAMA" ]; then
  llm_layers=16
  llm_dim=2048
fi

comment='RTSLLM-Weather'

accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_96 \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj 'one_cycle' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model --llm_dim "$llm_dim" \
  --llm_layers "$llm_layers"

accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_192 \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj 'one_cycle' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model --llm_dim "$llm_dim" \
  --llm_layers "$llm_layers"

accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_336 \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj 'one_cycle' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model --llm_dim "$llm_dim" \
  --llm_layers "$llm_layers"

accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_720 \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj 'one_cycle' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model --llm_dim "$llm_dim" \
  --llm_layers "$llm_layers"