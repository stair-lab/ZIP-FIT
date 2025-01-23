# af_tokenizers.py

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
        # 2) Get Tokenizer & BOS/EOS/PAD info
        # I often have this line and we should be able to see quickly what the tokenizers outputs for the models in question
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
            print(f'--> WARNING: NOTE THIS MODEL {repo} DIDNT HAVE A PAD TOKEN, WE ASSINGED IT TO EOS.')
        print(f'BOS/EOS/PAD --> BOS: {tokenizer.bos_token_id} EOS: {tokenizer.eos_token_id} PAD: {tokenizer.pad_token_id}')
        print(f'BOS/EOS/PAD --> BOS: {tokenizer.bos_token} EOS: {tokenizer.eos_token} PAD: {tokenizer.pad_token}')

        # 3) Tokenize according to common ways I do it for thi project
        print(f'Current raw rext (about to be tokenized): {text_examples}')

        # TFA encodings
        print(f'TFA Encoded text (explict args!): {tokenizer(text_examples, add_special_tokens=False)=}')
        
        # Train encodings
        # Note: add_eos_token is not always present, so we won't be using: tok = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", trust_remote_code=True, add_eos_token=True)
        print(f"Training (batch) Encoded text args (explict args!): {tokenizer(text_examples, padding='max_length', truncation=True, max_length=13, return_tensors='pt', padding_side='right')=}")
        print()


if __name__ == "__main__":
    main()