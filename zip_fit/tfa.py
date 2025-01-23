# tfa.py

import os
import time
import torch
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel
)
from datasets import load_dataset

def seed_everything(seed: int = 42):
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    """
    import random
    import numpy as np
    from transformers import set_seed as hf_set_seed

    print(f"Setting random seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        hf_set_seed(seed)
    else:
        print("Warning: Transformers is only fully deterministic on GPU")


def teacher_forced_accuracy_tfa(
    prompt: str,
    response: str,
    model: PreTrainedModel,
    repo: str,
    device: str = "cuda"
) -> float:
    """
    Teacher-forced accuracy (token-level) on `response` given a concatenated text = prompt + response.

    Steps:
      1) Combined text = prompt + "\n\n" + response
      2) Tokenize combined text => shape: (1, total_seq_len)
      3) Forward pass => logits shape: (1, total_seq_len, vocab_size)
      4) Identify the token range for the response
      5) Compare the predicted tokens in that range with the reference response tokens
      6) Return fraction matched in [0, 1]

    Notes about BOS/EOS/PAD:
      - Because we do per-example calls (prompt+response) only, no extra padding is needed.
      - If the model inserts BOS/EOS automatically, that is consistent for each example's TFA.
      - We do NOT forcibly add an EOS token; if you need that, append it manually to `response`.
    """

    # 1) Combine text
    combined_text = prompt + "\n\n" + response

    # 2) Use the tokenizer from the same `repo` to ensure consistency
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # 3) Tokenize entire reference
    enc = tokenizer(combined_text, return_tensors="pt")
    # shape: (1, total_seq_len)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape: (1, total_seq_len, vocab_size)
    preds = torch.argmax(logits, dim=-1)  # shape: (1, total_seq_len)

    # 4) Tokenize the response alone to find how many tokens it has
    response_enc = tokenizer(response, add_special_tokens=False)
    len_response = len(response_enc["input_ids"])

    # Tokenize the prompt alone for length
    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    len_prompt = len(prompt_enc["input_ids"])

    total_seq_len = input_ids.size(1)

    # If the combined text is too short or the response doesn't fit, skip
    if len_prompt + len_response >= total_seq_len:
        return 0.0

    # Teacher forcing alignment:
    #   model's position t attempts to predict token at position t+1
    pred_slice = preds[:, len_prompt : (len_prompt + len_response)]
    label_slice = input_ids[:, (len_prompt + 1) : (len_prompt + 1 + len_response)]

    if pred_slice.size(1) == 0 or label_slice.size(1) == 0:
        return 0.0

    correctness = (pred_slice == label_slice).float()  # shape: (1, number_of_response_tokens)
    acc = correctness.mean().item()
    return acc


def compute_tfa_for_subds(
    sub_ds,
    model: PreTrainedModel,
    repo: str,
    prompt_format_fn=None,
    device: str = "cuda"
) -> float:
    """
    Process an entire subset of data (sub_ds) and compute the average TFA across all examples.

    Parameters:
      sub_ds: The subset of the dataset (like a HuggingFace 'Dataset' slice).
      model:  A language model (transformers PreTrainedModel).
      repo:   The model repo string, used to load the correct tokenizer in teacher_forced_accuracy_tfa.
      prompt_format_fn: Optional function that transforms the raw 'nl_statement' into a 'prompt'.
      device: 'cuda' or 'cpu'.

    Returns:
      float: The average TFA over all examples in sub_ds.
    """
    sum_acc = 0.0
    count = 0

    for i, example in enumerate(sub_ds):
        nl_statement = example["nl_statement"]
        formal_statement = example["formal_statement"]

        if prompt_format_fn is not None:
            prompt = prompt_format_fn(nl_statement)
        else:
            # Default: straightforward instruction
            prompt = (
                "Translate the natural language version of the mathematical statement "
                f"to a formal Lean version:\n{nl_statement}\n"
            )

        acc_i = teacher_forced_accuracy_tfa(
            prompt=prompt,
            response=formal_statement,
            model=model,
            repo=repo,
            device=device
        )
        sum_acc += acc_i
        count += 1

        print(f" Example {i}: TFA = {acc_i:.4f}")

    return sum_acc / count if count > 0 else 0.0


def main():
    import time

    global_start_time = time.time()  # Start overall timer

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # choose GPU
    seed_everything()

    # 1) Load the ProofNet validation set
    ds = load_dataset("hoskinson-center/proofnet", split="validation")

    # We'll just do the first N examples for demonstration
    N = 5
    sub_ds = ds.select(range(min(N, len(ds))))

    # 2) Our model list (including all desired models, even if some remain commented)
    model_token_configs = [
        {
            "name": "internlm2-math-plus-1_8b",
            "repo": "internlm/internlm2-math-plus-1_8b",
        },
        {
            "name": "google/gemma-2-2b",
            "repo": "google/gemma-2-2b",
        },
        {
            "name": "Mistral-7B-v0.1",
            "repo": "mistralai/Mistral-7B-v0.1",
        },
        {
            "name": "google/codegemma-2b",
            "repo": "google/codegemma-2b",
        },
        {
            "name": "Meta-Llama-3-8B",
            "repo": "meta-llama/Meta-Llama-3-8B",
        },
        {
            "name": "Meta-Llama-3-8B-Instruct",
            "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        },
        {
            "name": "google/gemma-2-2b-it",
            "repo": "google/gemma-2-2b-it",
        },
        {
            "name": "GPT-2 (small)",
            "repo": "gpt2",
        },
    ]

    # Example of a custom prompt format function
    def my_prompt_format(nl_statement: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{nl_statement}\n"
        )

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
            prompt_format_fn=my_prompt_format,
            device=device
        )

        # End per-model timer
        model_end_time = time.time()
        model_seconds = model_end_time - model_start_time

        print(f" => Average TFA for {model_name} on these {N} example(s) = {avg_tfa:.4f}")
        print(f" => Time to compute TFA for {model_name}: {model_seconds:.2f} seconds.")

    # End overall timer
    global_end_time = time.time()
    total_seconds = global_end_time - global_start_time
    print(f"\nDone. Total run time for all models: {total_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()
