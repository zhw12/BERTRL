export DATASET="fb237"
export SUFFIX="_hop3_full_neg10_max_inductive"
CUDA_VISIBLE_DEVICES=0 python run_bertrl.py \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --data_dir ./bertrl_data/${DATASET}${SUFFIX} \
  --learning_rate 5e-5 \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --num_train_epochs 2.0 \
  --output_dir output_${DATASET}${SUFFIX} \
  --logging_steps 100 \
  --evaluation_strategy epoch \
  --overwrite_output_dir \
  --save_total_limit 1 \
  --evaluation_strategy epoch \
  --do_predict \
  --do_eval \
  --evaluate_during_training \
  --eval_steps 1000 \
  --save_steps 1000 \
  --per_device_eval_batch_size 750 \
  --overwrite_cache
  

#   --warmup_steps 20000 \
#   --save_steps 20000


