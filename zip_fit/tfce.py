# tfce.py

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
    Teacher-forced cross-entropy (TFCE), ignoring the prompt portion:
      => -log p(gold_response | prompt) 
         but we only compute CE over the gold_response portion.

    Steps:
      1) combined_text = prompt + "\n\n" + gold_response
      2) Tokenize with add_special_tokens=False => shape: (1, seq_len)
      3) Forward pass => logits => shape: (1, seq_len, vocab_size)
      4) We measure how many tokens are in prompt vs. gold_response by re-tokenizing them individually.
      5) Build labels from input_ids => set label=-100 for [0..len_prompt-1] and beyond [len_prompt+len_gold-1].
      6) Flatten => shape: (seq_len, vocab_size) for logits, shape: (seq_len,) for labels
      7) cross_entropy(..., ignore_index=-100, reduction=reduction)

    If `reduction='mean'`, we get the *average* CE across all gold-response tokens.
    If `reduction='sum'`, we get total negative log-likelihood across them.

    Notes:
      - We do not forcibly add BOS or EOS here. We skip it to match a "bare-bones" style,
        similar to the updated tfa.py that also ignores explicit BOS/EOS tokens.
      - If the combined text is truncated or too short, we return 0.0 as a fallback.

    Returns:
      float => final scalar cross-entropy or negative log-likelihood.
    """

    # 1) Load the tokenizer from the same repo
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # 2) Combine text
    combined_text = prompt + "\n\n" + gold_response

    # 3) Tokenize => shape => (1, seq_len)
    enc = tokenizer(
        combined_text,
        return_tensors="pt",
        add_special_tokens=False
    )
    input_ids = enc["input_ids"].to(device)  # shape => (1, seq_len)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)  # shape => (1, seq_len)

    batch_size, seq_len = input_ids.shape

    # 4) measure how many tokens in prompt vs. gold_response
    #    ignoring BOS/EOS
    enc_prompt = tokenizer(prompt, add_special_tokens=False)
    len_prompt = len(enc_prompt["input_ids"])

    enc_gold = tokenizer(gold_response, add_special_tokens=False)
    len_gold = len(enc_gold["input_ids"])

    # if truncated
    if len_prompt + len_gold > seq_len:
        return 0.0

    # 5) forward pass => logits => shape => (1, seq_len, vocab_size)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape => (1, seq_len, vocab_size)
    vocab_size = logits.size(-1)

    # build labels => a copy of input_ids
    labels = input_ids.clone()  # shape => (1, seq_len)
    # set label=-100 for [0..len_prompt-1] => ignoring the prompt portion
    labels[0, :len_prompt] = -100
    # also ignore any portion after the gold_response
    end_idx = len_prompt + len_gold
    if end_idx < seq_len:
        labels[0, end_idx:] = -100

    # 6) Flatten => shape => (seq_len, vocab_size) for logits, (seq_len,) for labels
    logits_2d = logits.view(-1, vocab_size)  # (seq_len, vocab_size)
    labels_1d = labels.view(-1)             # (seq_len,)

    # 7) cross_entropy => ignoring -100 => only gold_response portion
    loss = F.cross_entropy(
        logits_2d,
        labels_1d,
        ignore_index=-100,
        reduction=reduction
    )
    return float(loss.item())


def compute_loss_for_subds(
    sub_ds,
    model: PreTrainedModel,
    repo: str,
    prompt_format_fn=None,
    device: str = "cuda",
    N: Optional[int] = None,
    reduction: str = "mean"
) -> float:
    """
    Loops over up to N examples in sub_ds, computing loss_full_gold_ref(...) 
    for each example, then we average across all examples.

    If N is None or > len(sub_ds), we use all.
    If `reduction='mean'`, each example's cross-entropy is an average over tokens. 
    Then we average those per-example results => final scalar.

    If `reduction='sum'`, each example is total negative log-likelihood. 
    We then average those sums across examples => final float.

    sub_ds[i] is expected to have:
      'nl_statement': the prompt
      'formal_statement': the gold reference
    or adapt as needed.

    Returns: float in [0..∞).
    """
    size = len(sub_ds)
    if N is None or N > size:
        N = size

    sum_loss = 0.0
    for i in range(N):
        ex = sub_ds[i]
        nl_statement    = ex["nl_statement"]
        formal_statement= ex["formal_statement"]

        # build prompt if desired
        if prompt_format_fn is not None:
            prompt = prompt_format_fn(nl_statement)
        else:
            prompt = (f"Translate the statement to Lean:\n{nl_statement}\n")

        # measure
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
        return float('nan')  # or 0.0

    return sum_loss / N


def main():
    """
    Example main script:
      1) seed everything
      2) load 5 examples from proofnet
      3) evaluate cross-entropy ignoring prompt portion (teacher-forcing style)
         on multiple models
      4) print average results
      5) time it
    """
    start_time = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    seed_everything()

    ds = load_dataset("hoskinson-center/proofnet", split="validation")
    N = 5
    sub_ds = ds.select(range(min(N, len(ds))))

    # example model list
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

    for cfg in model_token_configs:
        name = cfg["name"]
        repo = cfg["repo"]
        print(f"\nEvaluating teacher-forced cross-entropy for {name} on {N} examples...")

        st = time.time()
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).to(device)

        avg_loss = compute_loss_for_subds(
            sub_ds=sub_ds,
            model=model,
            repo=repo,
            prompt_format_fn=my_prompt_format,
            device=device,
            N=N,
            reduction="mean"
        )

        ed = time.time()
        print(f" => Average TFCE for {name}: {avg_loss:.4f}")
        print(f" => Time: {ed - st:.2f} s")

    total = time.time() - start_time
    print(f"\nAll done. total script time={total:.2f}s")


if __name__ == "__main__":
    main()
