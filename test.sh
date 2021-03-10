
export DATASET="fb237"
export MODEL_SUFFIX="_hop3_full_neg10_path3_max_inductive"
export DATA_SUFFIX="_hop3_full_neg10_path3_max_inductive"
export OUTPUT_SUFFIX="_hop3_full_neg10_path3_max_inductive_new"
CUDA_VISIBLE_DEVICES=0  python run_bertrl.py \
  --model_name_or_path ./output_${DATASET}${MODEL_SUFFIX}/ \
  --task_name MRPC \
  --do_predict \
  --data_dir ./bertrl_data/${DATASET}${DATA_SUFFIX}\
  --max_seq_length 128 \
  --output_dir output_${DATASET}${OUTPUT_SUFFIX}/ \
  --per_device_eval_batch_size 1000 \
  --overwrite_output_dir \
  --overwrite_cache