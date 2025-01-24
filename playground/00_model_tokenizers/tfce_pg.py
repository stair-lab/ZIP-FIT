# tfce.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel



def loss_full_gold_ref(
    prompt: str,
    gold_response: str,
    model: PreTrainedModel,
    repo: str,
    device: str = "cuda",
    reduction: str = "mean"
) -> float:
    """
    Computes the negative log-likelihood (cross-entropy) of the 'gold_response'
    given 'prompt', ignoring the prompt portion in the loss. This is the standard
    "teacher forcing" style approach in a single forward pass, with no actual
    AR generation.

    Steps:
      1) combined_text = prompt + "\n\n" + gold_response
      2) Tokenize => shape: (1, seq_len)
      3) Forward pass => logits => shape: (1, seq_len, vocab_size)
      4) Build 'labels' = copy of input_ids, setting -100 for the prompt portion
         so that cross-entropy only applies to the gold response tokens
      5) cross_entropy(logits, labels, ignore_index=-100)

    If reduction="mean", returns the *average* cross-entropy per token in 
    the gold response. If reduction="sum", returns the total negative log-likelihood.
    
    This exactly matches your conversation's approach: no AR sampling, just a
    single pass with the entire (prompt+gold) input. 
    """

    # 1) Combine text
    combined_text = prompt + "\n\n" + gold_response

    # 2) Load tokenizer from the same repo to ensure matching tokenization
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # 3) Tokenize => shape: (1, seq_len)
    enc = tokenizer(combined_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # shape: (1, seq_len)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    seq_len = input_ids.size(1)

    # 4) Single forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # shape: (1, seq_len, vocab_size)
    logits = outputs.logits

    # We'll create labels = copy of input_ids, but set label=-100 for the prompt portion
    labels = input_ids.clone()

    # measure prompt length vs. gold_response length
    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    len_prompt = len(prompt_enc["input_ids"])

    response_enc = tokenizer(gold_response, add_special_tokens=False)
    len_response = len(response_enc["input_ids"])

    # If the combined text is truncated or too short
    if len_prompt + len_response >= seq_len:
        return 0.0

    #  - ignore the prompt portion => label = -100
    labels[0, :len_prompt] = -100

    #  - also ignore anything beyond the gold_response portion
    end_idx = len_prompt + len_response
    if end_idx < seq_len:
        labels[0, end_idx:] = -100

    # Flatten
    vocab_size = logits.size(-1)
    logits_2d = logits.view(-1, vocab_size)   # shape: (seq_len, vocab_size)
    labels_1d = labels.view(-1)              # shape: (seq_len,)

    # 5) Cross-entropy => ignore_index=-100 => ignoring prompt tokens
    loss = F.cross_entropy(
        logits_2d,
        labels_1d,
        ignore_index=-100,
        reduction=reduction
    )
    return loss.item()

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


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # choose GPU
    seed_everything()

    # 1) Our model list (including all desired models, even if some remain commented)
    model_token_configs = [
        # {
        #     "name": "internlm2-math-plus-1_8b",
        #     "repo": "internlm/internlm2-math-plus-1_8b",
        # },
        # {
        #     "name": "google/gemma-2-2b",
        #     "repo": "google/gemma-2-2b",
        # },
        {
            "name": "Mistral-7B-v0.1",
            "repo": "mistralai/Mistral-7B-v0.1",
        },
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print('\nLoop starting to see model tokenizations...')
    for config in model_token_configs:
        model_name = config["name"]
        repo = config["repo"]
        print(f'Model repo: {repo}')

        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).to(device)

        # putting the eos string token explicitly then letting the model pad, even if eos == pad, makes sure that we do train on the intended tokens (including eos).
        text_examples = [
            "Hello world.</s>",
            "The dog is brown.</s>"
        ]


if __name__ == "__main__":
    main()