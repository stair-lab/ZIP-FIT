#!/bin/bash

# Setup conda
source /lfs/skampere1/0/brando9/miniconda/etc/profile.d/conda.sh

# Setup model and paths
export model_name_or_path="Qwen/Qwen2.5-0.5B"
export model_args="pretrained=${model_name_or_path},dtype=auto"
export model_task_output_path='$HOME/data/runs_putnam_axiom_zipfit_outputs'
mkdir -p $model_task_output_path
export CUDA_VISIBLE_DEVICES=6

# Test train split
echo "Testing train split..."
conda activate zip_fit2
time lm_eval --model vllm \
    --model_args "${model_args}" \
    --tasks putnam_axiom_zipfit_train \
    --trust_remote_code \
    --batch_size auto:4 \
    --device cuda \
    --output_path ${model_task_output_path}/results_train.json \
    --log_samples

# Test validation split
echo "Testing validation split..."
conda activate zip_fit2
time lm_eval --model vllm \
    --model_args "${model_args}" \
    --tasks putnam_axiom_zipfit_val \
    --trust_remote_code \
    --batch_size auto:4 \
    --device cuda \
    --output_path ${model_task_output_path}/results_val.json \
    --log_samples

# Test test split
echo "Testing test split..."
conda activate zip_fit2
time lm_eval --model vllm \
    --model_args "${model_args}" \
    --tasks putnam_axiom_zipfit_test \
    --trust_remote_code \
    --batch_size auto:4 \
    --device cuda \
    --output_path ${model_task_output_path}/results_test.json \
    --log_samples 