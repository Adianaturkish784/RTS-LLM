model_name=RTSLLM
train_epochs=10
learning_rate=1e-4

master_port=00097
batch_size=32
d_model=32
d_ff=128

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

comment='RTSLLM-ETTm2'

accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'one_cycle' \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model --llm_dim "$llm_dim" \
  --llm_layers "$llm_layers"

accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
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
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
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
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj 'one_cycle' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model --llm_dim "$llm_dim" \
  --llm_layers "$llm_layers"