# loss_gold_ref_tfce.py

import os
import time
import math
import torch
import torch.nn.functional as F
from typing import Optional, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel
)
from datasets import load_dataset

def seed_everything(seed: int = 42):
    """
    Basic seeding for reproducibility across Python, NumPy, and PyTorch.
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


def loss_full_gold_ref(
    prompt: str,
    gold_response: str,
    model: PreTrainedModel,
    repo: str,
    device: str = "cuda",
    reduction: str = "mean"
) -> float:
    """
    Teacher-forced cross-entropy (TFCE) for the gold reference:
      => -log p(gold_response | prompt) ignoring the prompt portion in the loss.

    Steps:
      3) combined_text = (prompt + "\n\n" + gold_response)
      4) Tokenize with add_special_tokens=False => shape: (1, seq_len)
      5) Forward pass => logits => shape: (1, seq_len, vocab_size)
      6) labels = input_ids.clone(), set label=-100 for [0..len_prompt-1] and anything after gold_response
      7) Flatten => shape: (seq_len, vocab_size) for logits, shape: (seq_len,) for labels
      8) cross_entropy(..., ignore_index=-100, reduction=reduction)

    If reduction='mean', we get average CE over the solution tokens. 
    If 'sum', we get total negative log-likelihood across those tokens.

    This approach does NO generation and is consistent with the conversation: 
    “Just do a single forward pass on (prompt + gold reference).” 
    """

    # 1) Load the tokenizer from the same repo (ensures same vocabulary & special tokens).
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # 2) Combine text
    combined_text = prompt + "\n\n" + gold_response

    # 3) Tokenize with add_special_tokens=False => shape: (1, seq_len)
    enc = tokenizer(combined_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)  # shape: (1, seq_len)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)  # shape: (1, seq_len)

    batch_size = input_ids.size(0)   # Should be 1
    seq_len = input_ids.size(1)

    # 4) Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # shape: (1, seq_len, vocab_size)
    logits = outputs.logits
    vocab_size = logits.size(-1)

    # measure lengths for prompt vs. gold_response
    # (We do add_special_tokens=False so we see raw offsets.)
    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    len_prompt = len(prompt_enc["input_ids"])

    response_enc = tokenizer(gold_response, add_special_tokens=False)
    len_response = len(response_enc["input_ids"])

    # If the text is truncated or too short
    if len_prompt + len_response > seq_len:
        # fallback
        return 0.0

    # 5) labels = input_ids.clone() => shape: (1, seq_len)
    #    We do clone() so we can safely set -100 without modifying input_ids in-place.
    labels = input_ids.clone()

    # set label=-100 for the prompt portion
    labels[0, :len_prompt] = -100
    # also ignore anything beyond gold_response portion
    end_idx = len_prompt + len_response
    if end_idx < seq_len:
        labels[0, end_idx:] = -100

    # 6) Flatten => needed by F.cross_entropy, which expects (N, C) vs (N,)
    # shape: (1, seq_len, vocab_size) => (seq_len, vocab_size)
    logits_2d = logits.view(-1, vocab_size)   # shape: (seq_len, vocab_size)
    labels_1d = labels.view(-1)              # shape: (seq_len,)

    # 7) cross-entropy => ignore_index=-100 => ignoring the prompt tokens 
    loss = F.cross_entropy(
        logits_2d,
        labels_1d,
        ignore_index=-100,
        reduction=reduction
    )
    return loss.item()


def compute_loss_for_subds(
    sub_ds,
    model: PreTrainedModel,
    repo: str,
    prompt_format_fn=None,
    device: str = "cuda",
    N: int = None,
    reduction: str = "mean"
) -> float:
    """
    Loop over up to N examples from sub_ds, computing loss_full_gold_ref(...) 
    for each row, then average across examples. (An example-level average.)

    We assume sub_ds[i] has:
      "nl_statement": the prompt
      "formal_statement": the gold reference
    or adapt to your actual field names.

    If N is None or bigger than len(sub_ds), we use the entire sub_ds.

    If 'reduction' is "mean", each example's cross-entropy is an average over tokens,
    then we average across N examples => final float. 
    If "sum", each example is total negative log-likelihood, then we average over N examples.

    Return: float in [0..∞).
    """
    size = len(sub_ds)
    if N is None or N > size:
        N = size

    sum_loss = 0.0
    for i in range(N):
        row = sub_ds[i]
        # Adapt if your dataset keys differ
        nl_statement = row["nl_statement"]
        formal_statement = row["formal_statement"]

        if prompt_format_fn is not None:
            prompt = prompt_format_fn(nl_statement)
        else:
            prompt = f"Translate the statement to Lean:\n{nl_statement}\n"

        loss_i = loss_full_gold_ref(
            prompt=prompt,
            gold_response=formal_statement,
            model=model,
            repo=repo,
            device=device,
            reduction=reduction
        )
        sum_loss += loss_i

        print(f" Example {i}: loss={loss_i:.4f}")

    if N == 0:
        return float('nan')  # or 0.0 if you prefer

    avg_loss = sum_loss / N
    return avg_loss


def main():
    """
    Example main script:
      1) Seeds everything
      2) Loads a small portion of ProofNet's 'validation' set
      3) Loops over some models, computing the teacher-forced cross-entropy (TFCE)
         on the "formal_statement" portion, ignoring prompt tokens in the loss.
      4) Times each model's run.
    """
    start_time = time.time()

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # choose GPU
    seed_everything()

    # 2) Load a portion of proofnet 'validation'
    ds = load_dataset("hoskinson-center/proofnet", split="validation")
    N = 5
    sub_ds = ds.select(range(min(N, len(ds))))

    # 3) Model list
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
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def my_prompt_format(nl_statement: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{nl_statement}\n"
        )

    for config in model_token_configs:
        model_name = config["name"]
        repo = config["repo"]

        print(f"\nEvaluating teacher-forced cross-entropy for {model_name} on {N} examples...")

        model_start_time = time.time()

        # 4) load
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).to(device)

        # compute the average cross-entropy => example-level average
        avg_loss = compute_loss_for_subds(
            sub_ds=sub_ds,
            model=model,
            repo=repo,
            prompt_format_fn=my_prompt_format,
            device=device,
            N=N,
            reduction="mean"
        )

        model_end_time = time.time()
        secs = model_end_time - model_start_time

        print(f" => Average teacher-forced CE for {model_name}: {avg_loss:.4f}")
        print(f" => Time: {secs:.2f} sec")

    total_sec = time.time() - start_time
    print(f"\nAll done. Total run time: {total_sec:.2f} sec.")

if __name__ == "__main__":
    main()
