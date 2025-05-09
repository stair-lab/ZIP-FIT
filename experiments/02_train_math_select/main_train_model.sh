# - Args
export model_name="openai-community/gpt2"
# export model_name="Qwen/Qwen2.5-0.5B"
# export model_name="google/gemma-2-2b"
# export model_name="google/internlm2-math-plus-1_8b"
# export model_name="meta-llama/Llama-3.2-1B"
# export model_name="meta-llama/Llama-3.2-3B"
# since 3.1 8B gives issues with vvlm we are using 8B instead ref: https://github.com/vllm-project/vllm/issues/7382
# export model_name="meta-llama/Meta-Llama-3-8B"
# export model_name="meta-llama/Meta-Llama-3-8B-Instruct"
# export model_name="google/codegemma-2b"

export final_model_name="zipfit/model" # TODO

# - Training parameters - optimized to prevent OOM issues
export num_train_epochs=1
export do_eval=True
export eval_on_start=True
export evaluation_strategy="steps"
export eval_steps=50
export logging_steps=10
export per_device_train_batch_size=1
export gradient_accumulation_steps=8
export save_steps=100
export save_total_limit=1
export save_strategy="steps"
export bf16=True
export fp16=False
export optim="paged_adamw_32bit"
export learning_rate=1e-6
export weight_decay=1e-4
export gradient_checkpointing=True
export lr_scheduler_type="constant_with_warmup"
export warmup_ratio=0.05

export training_dataset_name="zipfit/math-select-06062025"
export training_split="train"
export training_eval_dataset_name="zipfit/Putnam-AXIOM-for-zip-fit-splits"
export training_eval_split="validation"

# - Run
export CUDA_VISIBLE_DEVICES=0
python -m zip_fit.nn_train.train \
  -model_name $model_name \
  -num_train_epochs $num_train_epochs \
  -do_eval $do_eval \
  -eval_on_start $eval_on_start \
  -evaluation_strategy $evaluation_strategy \
  -eval_steps $eval_steps \
  -logging_steps $logging_steps \
  -per_device_train_batch_size $per_device_train_batch_size \
  -gradient_accumulation_steps $gradient_accumulation_steps \
  -save_steps $save_steps \
  -save_total_limit $save_total_limit \
  -save_strategy $save_strategy \
  -bf16 $bf16 \
  -fp16 $fp16 \
  -optim $optim \
  -learning_rate $learning_rate \
  -weight_decay $weight_decay \
  -gradient_checkpointing $gradient_checkpointing \
  -lr_scheduler_type $lr_scheduler_type \
  -warmup_ratio $warmup_ratio \
  -final_model_name $final_model_name