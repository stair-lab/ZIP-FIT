import os

import torch

from zip_fit.metrics.tfa import compute_tfa_for_subds
from zip_fit.utils import seed_everything

def main():
    import time

    global_start_time = time.time()  # Start overall timer

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # choose GPU
    seed_everything()

    # 1) Load the ProofNet validation set
    ds = load_dataset("hoskinson-center/proofnet", split="validation")

    # Example of a custom prompt format function
    def my_prompt_format(prompt: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{prompt}\n"
        )
    ds = ds.map(lambda example: {'prompt': my_prompt_format(example['nl_statement']), 'gold_response': example['formal_statement']}, num_proc=24)

    # We'll just do the first N examples for demonstration
    N = 5
    sub_ds = ds.select(range(min(N, len(ds))))

    # 2) Our model list (including all desired models, even if some remain commented)
    model_token_configs = [
        # {
        #     "name": "internlm2-math-plus-1_8b",
        #     "repo": "internlm/internlm2-math-plus-1_8b",
        # },
        {
            "name": "google/gemma-2-2b",
            "repo": "google/gemma-2-2b",
        },
        # {
        #     "name": "Mistral-7B-v0.1",
        #     "repo": "mistralai/Mistral-7B-v0.1",
        # },
        # {
        #     "name": "google/codegemma-2b",
        #     "repo": "google/codegemma-2b",
        # },
        # {
        #     "name": "Meta-Llama-3-8B",
        #     "repo": "meta-llama/Meta-Llama-3-8B",
        # },
        # {
        #     "name": "Meta-Llama-3-8B-Instruct",
        #     "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        # },
        # {
        #     "name": "google/gemma-2-2b-it",
        #     "repo": "google/gemma-2-2b-it",
        # },
        # {
        #     "name": "GPT-2 (small)",
        #     "repo": "gpt2",
        # },
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for config in model_token_configs:
        model_name = config["name"]
        repo = config["repo"]

        print(f"\nEvaluating {model_name} from {repo} on {N} example(s) of ProofNet validation.")

        # Start per-model timer
        model_start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).to(device)
        avg_tfa = compute_tfa_for_subds(
            sub_ds=sub_ds,
            model=model,
            repo=repo,
            device=device
        )

        # End per-model timer
        model_end_time = time.time()
        model_seconds = model_end_time - model_start_time

        print(f" => Average TFA for {model_name} on these {N} example(s) = {avg_tfa:.4f}")
        print(f" => Time to compute TFA for {model_name}: {model_seconds:.2f} seconds.")

        # Test CallBack
        tfacb = TfaCallback(sub_ds, repo, 2, 2, 2)

    # End overall timer
    global_end_time = time.time()
    total_seconds = global_end_time - global_start_time
    print(f"\nDone. Total run time for all models: {total_seconds:.2f} seconds.")


def minimal_tfa_trainer_test():
    """
    A minimal script that demonstrates using the TfaCallback with 
    the Hugging Face Trainer for a tiny "toy" dataset. 
    It runs for 1 training step and triggers the TfaCallback logic 
    at training begin, evaluation, and training end.
    """
    from transformers import TrainingArguments, Trainer

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # choose GPU

    # 1) Basic seeding
    seed_everything(42)

    # 2) Load a small model (e.g. GPT-2).
    model_name = "gpt2"
    model_name = "google/gemma-2-2b"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # 3) Prepare dataset
    def my_prompt_format(prompt: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{prompt}\n"
        )
    ds_train = load_dataset("hoskinson-center/proofnet", split="validation")
    # ds_train = load_dataset("hoskinson-center/proofnet", split="test")
    ds_train = ds_train.with_format('torch')  
    ds_train = ds_train.map(
        lambda example: {
            'text': my_prompt_format(example['nl_statement']) 
                     + example['formal_statement'] 
                     + tokenizer.eos_token
        },
        num_proc=24
    )

    def tokenize_function(examples):
        # We create 'input_ids', 'attention_mask' and 'labels' = 'input_ids'
        tokenized = tokenizer(
            examples["text"], 
            padding='max_length', 
            max_length=300, 
            truncation=True
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    ds_train = ds_train.map(
        tokenize_function, 
        batched=True, 
        remove_columns=ds_train.column_names, 
        num_proc=24
    )

    ds_eval = load_dataset("hoskinson-center/proofnet", split="test")
    ds_eval = ds_eval.map(
        lambda ex: {
            'prompt': my_prompt_format(ex['nl_statement']), 
            'gold_response': ex['formal_statement']
        },
        num_proc=24
    )

    # 4) Minimal training args: run for 1 step, do evaluation at the same step.
    training_args = TrainingArguments(
       output_dir="./test-tfa-output",
       do_train=True,
       do_eval=True,
    #    max_steps=1,                # Only 1 step
       num_train_epochs=4,
       evaluation_strategy="steps",# Evaluate every 'eval_steps'
       eval_steps=1,               # so we'll evaluate after 1 step
       logging_steps=1,            # log after every step
       per_device_train_batch_size=4,
       save_strategy="no",
       # **FIX**: disable column pruning
       remove_unused_columns=False
    )

    # 5) Attach TfaCallback
    callback = TfaCallback(
        tfa_dataset=ds_eval,
        repo=model_name,
        n_begin=186,
        n_during=185,
        n_end=186
    )

    # 6) Build trainer
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=ds_train,
       eval_dataset=ds_train,  # or ds_eval, whichever you want for HF's standard .evaluate()
       callbacks=[callback]
    )

    # 7) Run training
    trainer.train()


if __name__ == "__main__":
    # main()
    minimal_tfa_trainer_test()