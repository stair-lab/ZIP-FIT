# - args
# export model_name="openai-community/gpt2"
# export model_name="qwen/qwen2.5-0.5b"
# export model_name="google/gemma-2-2b"
# export model_name="google/internlm2-math-plus-1_8b"
# export model_name="meta-llama/llama-3.2-1b"
# export model_name="meta-llama/llama-3.2-3b"
# since 3.1 8b gives issues with vvlm we are using 8b instead ref: https://github.com/vllm-project/vllm/issues/7382
# export model_name="meta-llama/meta-llama-3-8b"
# export model_name="meta-llama/meta-llama-3-8b-instruct"
# export model_name="meta-llama/meta-llama-3-8b-instruct"
# export model_name="google/codegemma-2b"

export model_name="internlm/internlm2-math-plus-1_8b"  

# export model_name="qwen/qwen3-8b"  

# export cuda_visible_devices=4
# export model_name="qwen/qwen3-14b" # doesn't fit in 1 gpu a100 80gb

# export model_name="qwen/qwq-32b"

# - training parameters - optimized to prevent oom issues
# real run
export max_steps=-1
export num_train_epochs=1

# debug ru
# export max_steps=1
# export num_train_epochs=-1

export do_eval=true
export eval_on_start=true
export eval_strategy="steps"
export eval_steps=25
export logging_steps=10
export per_device_train_batch_size=1
export gradient_accumulation_steps=8
export save_steps=100
export save_total_limit=1
export save_strategy="steps"
export bf16=true
export fp16=false
export optim="paged_adamw_32bit"
export learning_rate=1e-7
export weight_decay=1e-4
export gradient_checkpointing=true
export lr_scheduler_type="constant_with_warmup"
export warmup_ratio=0.05

export block_size=1024
# export block_size=800
# export block_size=512
# export block_size=400
# export block_size=256
# export block_size=2

export training_dataset_name="zipfit/math-select-06062025"
export training_split="src"

export training_dataset_name="UDACA/DSIR-train"
export training_split="train"

export training_dataset_name="UDACA/ZIPFIT-train"
export training_split="train"

export training_eval_dataset_name="zipfit/Putnam-AXIOM-for-zip-fit-splits"
export training_eval_split="test"

export training_tf_eval_dataset_name="zipfit/Putnam-AXIOM-for-zip-fit-splits"
export training_tf_eval_split="test"

export final_model_name="zipfit/math-select-mdl-${model_name#*/}-ds-${training_dataset_name#*/}"

export mode="dryrun"
export mode="online"

# - run
export cuda_visible_devices=1
export cuda_visible_devices=2
# export cuda_visible_devices=7
conda activate zip_fit
python -m zip_fit.nn_train.train \
  -model_name $model_name \
  -num_train_epochs $num_train_epochs \
  -do_eval $do_eval \
  -eval_on_start $eval_on_start \
  -eval_strategy $eval_strategy \
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
  -final_model_name $final_model_name \
  -max_steps $max_steps \
  -training_dataset_name $training_dataset_name \
  -training_split $training_split \
  -training_eval_dataset_name $training_eval_dataset_name \
  -training_eval_split $training_eval_split \
  -training_tf_eval_dataset_name $training_tf_eval_dataset_name \
  -training_tf_eval_split $training_tf_eval_split \
  -mode $mode \
  -block_size $block_size