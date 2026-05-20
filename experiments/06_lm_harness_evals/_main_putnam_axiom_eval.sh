# - [Optional] Setup
git clone git@github.com:brando90/lm-evaluation-harness.git
cd lm-evaluation-harness

conda create -n zip_fit2 python=3.11
conda activate zip_fit2
pip install --upgrade pip

pip install -e ~/ZIP-FIT
pip install -e ~/lm-evaluation-harness

# model & model args
# export model_name_or_path="Qwen/Qwen2.5-0.5B"
export model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
export model_args="pretrained=${model_name_or_path},dtype=auto"

# Available Putnam Axiom tasks (choose one):
# - putnam_axiom_original: Uses the 'full_eval' split (522 original problems, variation=0)
# - putnam_axiom_variations: Uses the 'variations' split (500 variations, variation=1)
# - putnam_axiom_variations_org: Uses the 'originals_for_generating_vars' split (100 original problems used for creating variations, variation=0)
export tasks="putnam_axiom_original"
export tasks="putnam_axiom_variations"
# export tasks="putnam_axiom_variations_org"

# saving outputs
export model_task_output_path='$HOME/data/runs_putnam_axiom_outputs'
mkdir -p $model_task_output_path

# choose gpu
export CUDA_VISIBLE_DEVICES=6

# run lm eval with putnam-axiom
conda activate zip_fit2
time lm_eval --model vllm \
    --model_args "${model_args}" \
    --tasks ${tasks} \
    --trust_remote_code \
    --batch_size auto:4 \
    --device cuda \
    --output_path ${model_task_output_path}/results_${tasks}.json \
    --log_samples
