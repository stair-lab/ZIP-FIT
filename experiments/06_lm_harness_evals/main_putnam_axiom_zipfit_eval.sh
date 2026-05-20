#!/bin/bash
# Main script for running Putnam AXIOM ZIP-FIT evaluations

# - [Optional] Setup
# git clone git@github.com:brando90/lm-evaluation-harness.git -b zip_fit
# cd lm-evaluation-harness
# 
# conda create -n zip_fit2 python=3.11
# conda activate zip_fit2
# pip install --upgrade pip
# 
# pip install -e ~/ZIP-FIT
# pip install -e ~/lm-evaluation-harness

export CUDA_VISIBLE_DEVICES=2
# export model_name="internlm/internlm2-math-plus-1_8b"
export CUDA_VISIBLE_DEVICES=1
export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d13_t13h_51m_30s/zipfit/math-select-mdl-internlm2-math-plus-1_8b-ds-math-select-06062025-2025_m05_d13_t13h_51m_30"
export CUDA_VISIBLE_DEVICES=1
export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d13_t15h_32m_47s/zipfit/math-select-mdl-internlm2-math-plus-1_8b-ds-DSIR-train-2025_m05_d13_t15h_32m_47s"
export CUDA_VISIBLE_DEVICES=7
export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d13_t15h_50m_53s/zipfit/math-select-mdl-internlm2-math-plus-1_8b-ds-ZIPFIT-train-2025_m05_d13_t15h_50m_53s"

# export CUDA_VISIBLE_DEVICES=4
# export model_name="meta-llama/Meta-Llama-3-8B-Instruct"
# export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d10_t23h_34m_59s/zipfit/math-select-mdl-Meta-Llama-3-8B-Instruct-ds-math-select-06062025-2025_m05_d10_t23h_34m_59s"
# export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d10_t23h_47m_45s/zipfit/math-select-mdl-Meta-Llama-3-8B-Instruct-ds-DSIR-train-2025_m05_d10_t23h_47m_45s"
# export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d10_t23h_49m_01s/zipfit/math-select-mdl-Meta-Llama-3-8B-Instruct-ds-ZIPFIT-train-2025_m05_d10_t23h_49m_01s"

# export CUDA_VISIBLE_DEVICES=2
# export model_name="meta-llama/Meta-Llama-3-8B"
# export CUDA_VISIBLE_DEVICES=1
# export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d10_t23h_57m_54s/zipfit/math-select-mdl-Meta-Llama-3-8B-ds-math-select-06062025-2025_m05_d10_t23h_57m_54s"
# export CUDA_VISIBLE_DEVICES=1
# export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d10_t23h_58m_25s/zipfit/math-select-mdl-Meta-Llama-3-8B-ds-DSIR-train-2025_m05_d10_t23h_58m_25s"
# export CUDA_VISIBLE_DEVICES=7
# export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d10_t23h_58m_59s/zipfit/math-select-mdl-Meta-Llama-3-8B-ds-ZIPFIT-train-2025_m05_d10_t23h_58m_59s"

# export CUDA_VISIBLE_DEVICES=4
# export model_name="Qwen/Qwen2.5-0.5B"

# export CUDA_VISIBLE_DEVICES=2
# export model_name="Qwen/Qwen3-8B"  
# export CUDA_VISIBLE_DEVICES=2
# export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d13_t13h_38m_56s/zipfit/math-select-mdl-Qwen3-8B-ds-DSIR-train-2025_m05_d13_t13h_38m_56s"  
# export CUDA_VISIBLE_DEVICES=2
# export model_name="/lfs/skampere1/0/brando9/data/zipfit_less_runs/tfa_output_2025_m05_d13_t13h_40m_09s/zipfit/math-select-mdl-Qwen3-8B-ds-ZIPFIT-train-2025_m05_d13_t13h_40m_09s"

# export CUDA_VISIBLE_DEVICES=4
# export model_name="Qwen/Qwen3-14B"  

# export CUDA_VISIBLE_DEVICES=4
# export model_name="Qwen/QwQ-32B" 

export model_args="pretrained=${model_name},dtype=auto"

# Available Putnam Axiom ZIP-FIT tasks:
# - putnam_axiom_zipfit_train: Uses the 'train' split (150 problems)
# - putnam_axiom_zipfit_val: Uses the 'validation' split (150 problems)
# - putnam_axiom_zipfit_test: Uses the 'test' split (222 problems)
# - putnam_axiom_zipfit_all: Runs all splits together
export tasks="putnam_axiom_zipfit_test"
# export tasks="putnam_axiom_zipfit_all"

# saving outputs
export model_task_output_path='$HOME/data/runs_putnam_axiom_zipfit_outputs'
mkdir -p $model_task_output_path

# choose gpu
# run lm eval with putnam-axiom zipfit
cd ~/lm-evaluation-harness
conda activate zip_fit2
time lm_eval --model vllm \
    --model_args "${model_args}" \
    --tasks ${tasks} \
    --trust_remote_code \
    --batch_size auto:4 \
    --device cuda \
    --output_path ${model_task_output_path}/results_${tasks}.json \
    --log_samples 