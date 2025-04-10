#!/usr/bin/env python

import os
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from typing import List

def seed_everything(seed: int = 42):
    import random
    import numpy as np
    from transformers import set_seed as hf_set_seed

    print(f"Setting random seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)

def vectorized_loss_full_gold_ref(
    prompts: List[str],
    gold_responses: List[str],
    model: AutoModelForCausalLM,
    repo: str,
    device: str = "cuda",
    reduction: str = "mean"
) -> float:
    """
    Implementation from above
    """
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    B = len(prompts)
    assert B == len(gold_responses), "Mismatch: #prompts != #gold_responses"

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # Possibly add BOS/EOS
    processed_prompts = []
    processed_responses = []
    for i in range(B):
        p = prompts[i]
        r = gold_responses[i]
        if tokenizer.bos_token and not p.strip().startswith(tokenizer.bos_token):
            p = tokenizer.bos_token + p
        if tokenizer.eos_token and not r.strip().endswith(tokenizer.eos_token):
            r = r + tokenizer.eos_token

        processed_prompts.append(p)
        processed_responses.append(r)

    # combine
    combined_texts = []
    for i in range(B):
        combined_texts.append(processed_prompts[i] + "\n\n" + processed_responses[i])

    # tokenize all at once
    enc = tokenizer(
        combined_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    B2, max_seq_len = input_ids.shape
    assert B == B2, f"Expected B={B}, but got B2={B2}"

    # compute individual prompt/response lengths
    prompt_lens = []
    response_lens = []
    for i in range(B):
        p_enc = tokenizer(processed_prompts[i], add_special_tokens=False)
        r_enc = tokenizer(processed_responses[i], add_special_tokens=False)
        prompt_lens.append(len(p_enc["input_ids"]))
        response_lens.append(len(r_enc["input_ids"]))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape: (B, max_seq_len, vocab_size)
    vocab_size = logits.size(-1)

    # check for truncation
    for i in range(B):
        if prompt_lens[i] + response_lens[i] > max_seq_len:
            raise ValueError(
                f"Example {i}: truncated! prompt+resp length = {prompt_lens[i] + response_lens[i]} > max_seq_len={max_seq_len}"
            )

    # build labels
    labels = input_ids.clone()  # shape: (B, max_seq_len)
    for i in range(B):
        p_len = prompt_lens[i]
        r_len = response_lens[i]
        labels[i, :p_len] = -100
        end_idx = p_len + r_len
        if end_idx < max_seq_len:
            labels[i, end_idx:] = -100

    # flatten => shape: (B*max_seq_len, vocab_size) vs (B*max_seq_len,)
    logits_2d = logits.view(-1, vocab_size)
    labels_1d = labels.view(-1)

    loss = F.cross_entropy(
        logits_2d,
        labels_1d,
        ignore_index=-100,
        reduction=reduction
    )
    return loss.item()

def main():
    start_time = time.time()

    seed_everything()

    # load dataset
    ds = load_dataset("hoskinson-center/proofnet", split="validation")
    # pick e.g. 5 examples
    N = 5
    subset = ds.select(range(min(N, len(ds))))

    # build lists of prompts & gold_responses
    # We assume each row has "nl_statement", "formal_statement"
    prompts = []
    gold_responses = []
    for ex in subset:
        prompts.append(ex["nl_statement"])
        gold_responses.append(ex["formal_statement"])

    # model list
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

    for config in model_token_configs:
        model_name = config["name"]
        repo = config["repo"]

        print(f"\nComputing vectorized TFCE for {model_name} on {N} examples...")
        model_start = time.time()

        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).to(device)

        # do one forward pass for all examples
        tfce_loss = vectorized_loss_full_gold_ref(
            prompts=prompts,
            gold_responses=gold_responses,
            model=model,
            repo=repo,
            device=device,
            reduction="mean"
        )

        model_end = time.time()
        secs = model_end - model_start
        print(f" => Teacher-forced cross-entropy (mean) = {tfce_loss:.4f}")
        print(f" => Time: {secs:.2f}s")

    total_time = time.time() - start_time
    print(f"\nDone. Total run time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
