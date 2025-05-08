export seed=42
export k=5
export N=6
export model_name="gpt2"
# export model_name="Qwen/Qwen2.5-0.5B"
# export model_name="google/gemma-2-2b"
# export model_name="google/internlm2-math-plus-1_8b"
# export model_name="Meta-Llama-3-8B"
# export model_name="google/codegemma-2b"

python -m zip_fit.main.main_train_eval_lean4 \
    --seed $seed \
    --k $k \
    --N $N \
    --model_name $model_name
