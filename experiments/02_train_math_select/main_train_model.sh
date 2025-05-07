

# - Args
# export model_name="openai-community/gpt2"
# export model_name="Qwen/Qwen2.5-0.5B"
# export model_name="google/gemma-2-2b"
# export model_name="google/internlm2-math-plus-1_8b"
# export model_name="meta-llama/Llama-3.2-1B"
# export model_name="meta-llama/Llama-3.2-3B"
# since 3.1 8B gives issues with vvlm we are using 8B instead ref: https://github.com/vllm-project/vllm/issues/7382
# export model_name="meta-llama/Meta-Llama-3-8B"
export model_name="meta-llama/Meta-Llama-3-8B-Instruct"
# export model_name="google/codegemma-2b"

# - Run
export CUDA_VISIBLE_DEVICES=0
python -m zip_fit.nn_train.train