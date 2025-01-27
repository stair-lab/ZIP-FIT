#!/usr/bin/env python

# tfce.py

import os
import time
import random
import torch
from typing import Optional, Callable, Union
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, Dataset
import wandb

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


def loss_full_gold_ref(
    prompt: str,
    gold_response: str,
    model: PreTrainedModel,
    repo: str,
    device: str = "cuda",
    reduction: str = "mean"
) -> float:
    """
    Teacher-forced cross-entropy (TFCE), ignoring the prompt portion.
    
    => -log p(gold_response | prompt), but only computing CE over the gold_response tokens.

    Steps:
      1) combined_text = prompt + "\n\n" + gold_response
      2) Tokenize => shape (1, seq_len) with add_special_tokens=False
      3) Forward pass => logits shape (1, seq_len, vocab_size)
      4) Re-tokenize prompt/gold_response to measure lengths (len_prompt, len_gold)
      5) Build labels from input_ids => set label = -100 for [0..len_prompt-1] and beyond [len_prompt+len_gold-1]
      6) Flatten => cross_entropy(..., ignore_index=-100, reduction=reduction)
      7) Return float scalar (mean or sum over gold_response tokens).

    If text is truncated or too short, we return 0.0 as fallback.

    We do not forcibly add BOS/EOS, matching a "bare-bones" style (like your TFA code).
    """

    # 1) Load the tokenizer from the same repo
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # 2) Combine text
    combined_text = prompt + "\n\n" + gold_response

    # 3) Tokenize => shape (1, seq_len)
    enc = tokenizer(
        combined_text,
        return_tensors="pt",
        add_special_tokens=False
    )
    input_ids = enc["input_ids"].to(device)  # shape => (1, seq_len)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)  # shape => (1, seq_len)

    # measure lengths
    batch_size, seq_len = input_ids.shape

    # Re-tokenize prompt / gold_response to measure how many tokens each has
    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    len_prompt = len(prompt_enc["input_ids"])

    gold_enc = tokenizer(gold_response, add_special_tokens=False)
    len_gold = len(gold_enc["input_ids"])

    # If truncated or too short
    if (len_prompt + len_gold) > seq_len:
        return 0.0

    # 4) forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # shape => (1, seq_len, vocab_size)
    logits = outputs.logits
    vocab_size = logits.size(-1)

    # 5) Build labels from input_ids
    labels = input_ids.clone()  # (1, seq_len)
    # ignore the prompt portion
    labels[0, :len_prompt] = -100
    # ignore anything beyond gold_response portion
    end_idx = len_prompt + len_gold
    if end_idx < seq_len:
        labels[0, end_idx:] = -100

    # 6) Flatten
    logits_2d = logits.view(-1, vocab_size)  # (seq_len, vocab_size)
    labels_1d = labels.view(-1)             # (seq_len,)

    # 7) cross_entropy => ignoring -100
    ce_loss = F.cross_entropy(
        logits_2d,
        labels_1d,
        ignore_index=-100,
        reduction=reduction
    )
    return float(ce_loss.item())


def compute_tfce_for_subds(
    sub_ds: Dataset,
    model: PreTrainedModel,
    repo: str,
    device: str = "cuda",
    reduction: str = "mean",
    debug: bool = False
) -> float:
    """
    Process an entire subset of data (sub_ds) and compute the average TFCE across all examples.

    For each example, we do `loss_full_gold_ref(...)`. We average over examples.

    sub_ds[i] must have 'prompt' and 'gold_response'.
    If `debug=True`, we print each example's cross-entropy.
    """
    sum_loss = 0.0
    count = 0

    for i, example in enumerate(sub_ds):
        prompt = example["prompt"]
        gold_response = example["gold_response"]

        loss_i = loss_full_gold_ref(
            prompt=prompt,
            gold_response=gold_response,
            model=model,
            repo=repo,
            device=device,
            reduction=reduction
        )
        sum_loss += loss_i
        count += 1

        if debug:
            print(f" Example {i}: TFCE loss={loss_i:.4f}")

    return sum_loss / count if count > 0 else 0.0


class TfceCallback(TrainerCallback):
    """
    A callback that performs Teacher-Forced Cross-Entropy (TFCE) evaluations at:
      - on_train_begin => up to n_begin samples
      - on_evaluate    => up to n_during samples
      - on_train_end   => up to n_end samples (or entire set if n_end == -1)
    
    Very similar to TfaCallback but uses cross-entropy (loss_full_gold_ref).
    """

    def __init__(
        self,
        tfce_dataset: Dataset,
        repo: str,
        n_begin: int = -1,
        n_during: int = 2,
        n_end: int = -1,
        device: str = "cuda",
        reduction: str = "mean"
    ):
        """
        Args:
          tfce_dataset (Dataset):
            A dataset with 'prompt' & 'gold_response' columns for computing TFCE.

          repo (str):
            HF repo string for tokenization.

          n_begin (int):
            # examples for TFCE at train start. If 0 => skip. If -1 => entire dataset.

          n_during (int):
            # examples for TFCE at on_evaluate. If 0 => skip. If -1 => entire dataset.

          n_end (int):
            # examples for TFCE at train end. If 0 => skip. If -1 => entire dataset.

          device (str):
            "cuda" or "cpu".

          reduction (str):
            "mean" or "sum" for cross-entropy reduction.
        """
        super().__init__()
        self.tfce_dataset = tfce_dataset
        self.repo = repo
        self.n_begin = n_begin
        self.n_during = n_during
        self.n_end = n_end
        self.reduction = reduction

    def on_train_begin(self, args: TrainerState, state: TrainerState, control: TrainerControl, **kwargs):
        if self.n_begin == 0:
            return

        # if n_begin == -1 => entire dataset
        n = len(self.tfce_dataset) if self.n_begin == -1 else self.n_begin
        self._eval_tfce_and_log(n_samples=n, label="train_begin", state=state, **kwargs)

    def on_evaluate(self, args: TrainerState, state: TrainerState, control: TrainerControl, **kwargs):
        if self.n_during == 0:
            return

        # if n_during == -1 => entire dataset
        n = len(self.tfce_dataset) if self.n_during == -1 else self.n_during
        self._eval_tfce_and_log(n_samples=n, label="during_eval", state=state, **kwargs)

    def on_train_end(self, args: TrainerState, state: TrainerState, control: TrainerControl, **kwargs):
        if self.n_end == 0:
            return

        # if n_end == -1 => entire dataset
        n = len(self.tfce_dataset) if self.n_end == -1 else self.n_end
        self._eval_tfce_and_log(n_samples=n, label="train_end", state=state, **kwargs)

    def _eval_tfce_and_log(self, n_samples: int, label: str, state: TrainerState, **kwargs):
        """
        Helper that randomly samples up to n_samples from self.tfce_dataset,
        calls compute_tfce_for_subds, logs the result.
        """
        model = kwargs["model"]
        current_step = state.global_step
        device = next(model.parameters()).device

        ds_size = len(self.tfce_dataset)
        if ds_size == 0:
            return
        if n_samples > ds_size:
            n_samples = ds_size

        indices = random.sample(range(ds_size), k=n_samples)
        sub_ds = self.tfce_dataset.select(indices)

        tfce_val = compute_tfce_for_subds(
            sub_ds=sub_ds,
            model=model,
            repo=self.repo,
            device=device,
            reduction=self.reduction
        )
        log_dict = {f"tfce/{label}": tfce_val, "global_step": current_step}
        print(log_dict)
        wandb.log(log_dict)


def main():
    """
    Like the TFA main: 
      1) seed
      2) load a small portion of proofnet
      3) measure TFCE on some model(s)
      4) prints average results
    """
    start_time = time.time()
    seed_everything(42)

    # 1) load 5 examples from proofnet
    ds = load_dataset("hoskinson-center/proofnet", split="validation")
    def my_prompt_format(prompt: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{prompt}\n"
        )
    ds = ds.map(lambda ex: {'prompt': my_prompt_format(ex['nl_statement']), 'gold_response': ex['formal_statement']}, num_proc=24)

    N = 5
    sub_ds = ds.select(range(min(N, len(ds))))

    # 2) some models to test
    model_token_configs = [
        {
            "name": "google/gemma-2-2b",
            "repo": "google/gemma-2-2b",
        },
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3) measure TFCE
    for cfg in model_token_configs:
        name = cfg["name"]
        repo = cfg["repo"]
        print(f"\nEvaluating TFCE for {name} on {N} examples...")

        st = time.time()
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).to(device)

        avg_loss = compute_tfce_for_subds(
            sub_ds=sub_ds,
            model=model,
            repo=repo,
            device=device,
            reduction="mean"
        )
        ed = time.time()

        print(f" => Average TFCE for {name}: {avg_loss:.4f}")
        print(f" => Time: {(ed - st):.2f} s")

    print(f"\nAll done. total time = {time.time() - start_time:.2f} s")


def minimal_tfce_trainer_test():
    """
    A minimal script that demonstrates using the TfceCallback with 
    the Hugging Face Trainer for a tiny "toy" dataset. 
    It runs for 1 training step and triggers the TfceCallback logic 
    at training begin, evaluation, and train end.
    """
    from transformers import TrainingArguments, Trainer

    seed_everything(42)

    # 1) Load a small model (like GPT-2)
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # 2) Prepare a dataset for training
    def my_prompt_format(prompt: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{prompt}\n"
        )

    ds_train = load_dataset("hoskinson-center/proofnet", split="validation")
    ds_train = ds_train.map(lambda ex: {
        "text": my_prompt_format(ex["nl_statement"])
                 + ex["formal_statement"]
                 + tokenizer.eos_token
    }, num_proc=24)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            padding="max_length", 
            max_length=128, 
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

    # 3) Prepare a dataset for TFCE
    ds_eval = load_dataset("hoskinson-center/proofnet", split="test")
    ds_eval = ds_eval.map(lambda ex: {
        "prompt": my_prompt_format(ex["nl_statement"]),
        "gold_response": ex["formal_statement"]
    }, num_proc=24)

    # 4) minimal training args
    training_args = TrainingArguments(
        output_dir="./test-tfce-output",
        do_train=True,
        do_eval=True,
        max_steps=1,
        evaluation_strategy="steps",
        eval_steps=1,
        logging_steps=1,
        per_device_train_batch_size=2,
        remove_unused_columns=False,  # ensure prompt/gold_response remain accessible
        save_strategy="no"
    )

    # 5) attach TfceCallback
    callback = TfceCallback(
        tfce_dataset=ds_eval,
        repo=model_name,
        n_begin=2,   # 2 samples at train begin
        n_during=1,  # 1 sample at each on_evaluate
        n_end=2,     # 2 samples at train end
        device="cuda",
        reduction="mean"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_train,
        callbacks=[callback]
    )

    # 6) train
    trainer.train()


if __name__ == "__main__":
    # main()  # or test the direct usage
    minimal_tfce_trainer_test()
