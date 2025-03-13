#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the cross-entropy loss (average negative log-likelihood) of the
AI4M/gemma2-2b-gpt4-more-5-epochs model on the UDACA/proofnet-v3-lean4 dataset,
using the same "block grouping" logic used in training.

This ensures our evaluation matches the tokenization/block-packing approach
that was used in training.

Requires:
    - torch
    - transformers
    - datasets

Example usage:
    python evaluate_blocked_ce.py
"""

from typing import Dict, List, Optional
import os
import torch
from itertools import chain
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
)
################################################################################
# 1) Utility function: seed_everything()
################################################################################


def seed_everything(seed: int = 42) -> None:
    """
    Seed Python, NumPy, PyTorch, and (if applicable) transformers to
    achieve deterministic or reproducible results (where possible).
    """
    import random
    import numpy as np
    import torch
    from transformers import set_seed as hf_set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Some deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hf_set_seed(seed)  # For certain aspects of huggingface transformers


################################################################################
# 2) Utility function: tokenize_and_group_texts_via_blocks()
################################################################################


def tokenize_and_group_texts_via_blocks(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a batch of raw text examples and groups the tokens into fixed-size blocks,
    matching a typical "causal language modeling with block packing" approach.

    Steps:
      1. Prepend BOS token (if the tokenizer has one) and append EOS token to each example.
      2. Concatenate all tokenized results into one long list of token IDs.
      3. Truncate so that total length is an exact multiple of block_size.
      4. Slice into blocks of length block_size.
      5. Create 'input_ids' and 'labels' as PyTorch tensors, each of shape (num_blocks, block_size).

    Args:
        examples (Dict[str, List[str]]):
            A batch of examples from the dataset. Must contain a "text" key with a list of strings.
        tokenizer (PreTrainedTokenizerBase):
            Tokenizer used to tokenize each string.
        block_size (int):
            Target block size for the final grouped sequences.

    Returns:
        Dict[str, torch.Tensor]:
            A dictionary with:
              - "input_ids": shape (num_blocks, block_size)
              - "labels": shape (num_blocks, block_size)

            (No 'attention_mask' is returned here; for causal LM, it’s often optional.)
    """
    # Retrieve special token IDs (BOS/EOS). May be None if the tokenizer does not define them.
    eos_id: Optional[int] = tokenizer.eos_token_id
    bos_id: Optional[int] = tokenizer.bos_token_id

    # 1) Per-example tokenization with optional BOS and EOS tokens
    # "examples['text']" is a list of strings. We individually tokenize them,
    # then add the tokens to a master list for concatenation.
    batch_token_lists: List[List[int]] = []
    for txt in examples["text"]:
        # Prepend BOS if available
        prefix = [bos_id] if bos_id is not None else []
        # Tokenize (don't add special tokens automatically, so add_special_tokens=False if needed)
        enc = tokenizer(txt, add_special_tokens=False)
        # Append EOS if available
        suffix = [eos_id] if eos_id is not None else []

        token_ids = prefix + enc["input_ids"] + suffix
        batch_token_lists.append(token_ids)

    # 2) Concatenate into a single list
    concatenated: List[int] = list(chain(*batch_token_lists))

    # 3) Truncate to multiple of block_size
    total_length = len(concatenated)
    total_length = (total_length // block_size) * block_size
    truncated = concatenated[:total_length]

    # 4) Slice into blocks
    blocks: List[List[int]] = [
        truncated[i : i + block_size] for i in range(0, total_length, block_size)
    ]

    # 5) Create 'input_ids' and 'labels' (copy of input_ids) as PyTorch tensors
    # shape: (num_blocks, block_size)
    import torch

    if len(blocks) == 0:
        # Return empty Tensors if there's no data
        return {
            "input_ids": torch.empty((0, block_size), dtype=torch.long),
            "labels": torch.empty((0, block_size), dtype=torch.long),
        }

    input_ids_tensor = torch.tensor(blocks, dtype=torch.long)
    labels_tensor = input_ids_tensor.clone()

    return {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
    }


################################################################################
# 3) Utility function: compute_dataset_crossentropy_via_blocks()
################################################################################


def compute_dataset_crossentropy_via_blocks(
    tokenized_dataset: Dataset,
    model: AutoModelForCausalLM,
    batch_size: int = 2,
    device: str = "cuda",
) -> float:
    """
    Compute the average cross-entropy on a dataset that has already been
    grouped into blocks of size 'block_size'.

    The dataset should have columns 'input_ids' and 'labels', each shaped
    like (some_number_of_examples, block_size).

    We sum up loss * number_of_tokens, then divide by total_tokens.

    Args:
        tokenized_dataset (Dataset):
            The dataset after 'tokenize_and_group_texts_via_blocks'.
            Each row is one block. It has 'input_ids' and 'labels'.
        model (AutoModelForCausalLM):
            The language model in evaluation mode.
        batch_size (int):
            Batch size for evaluation.
        device (str):
            "cuda" if GPU available, otherwise "cpu".

    Returns:
        float: The cross-entropy in natural log (average negative log likelihood).
    """
    import torch
    from torch.utils.data import DataLoader

    model.to(device)
    model.eval()

    # Make a PyTorch DataLoader from the HF dataset
    # We only need the columns [input_ids, labels].
    def collate_fn(batch):
        # Each item in 'batch' is a dictionary with 'input_ids' and 'labels'.
        # Stack them.
        input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
        labels = torch.stack([x["labels"] for x in batch], dim=0)
        # (batch_size, block_size) shape
        return {
            "input_ids": input_ids.to(device),
            "labels": labels.to(device),
        }

    loader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            # batch["input_ids"] : shape [B, block_size]
            # batch["labels"]    : shape [B, block_size]
            out = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
            )
            # out.loss is the average cross-entropy over the entire batch, in natural log
            # i.e., sum over tokens / total tokens in the batch => average
            loss_value = out.loss.item()
            # Count how many tokens were in this batch
            # Since all are block_size, #tokens = B * block_size
            # but note that 'attention_mask' is not used here. If needed, 
            # you could do a more precise count. We'll assume all tokens are valid.
            bsz, block_sz = batch["input_ids"].size()
            num_tokens = bsz * block_sz

            total_loss += loss_value * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("nan")

    return total_loss / total_tokens


################################################################################
# 4) Main Script: Evaluate cross-entropy of "AI4M/gemma2-2b-gpt4-more-5-epochs"
################################################################################


def main() -> None:
    """
    Main function:
      1. Seeds everything for reproducibility.
      2. Loads the UDACA/proofnet-v3-lean4 dataset (validation split).
      3. Renames the text column if necessary (e.g., 'nl_statement' -> 'text').
      4. Tokenizes and groups text into blocks via "tokenize_and_group_texts_via_blocks".
      5. Computes cross-entropy via "compute_dataset_crossentropy_via_blocks".
      6. Prints the result.

    This replicates the block-grouping approach that was used in training code,
    making the evaluation consistent with how the model was trained.
    """

    seed_everything(42)

    # 1) Load dataset
    dataset_name = "UDACA/proofnet-v3-lean4"
    split_name = "validation"
    print(f"Loading dataset {dataset_name} [split={split_name}] ...")
    ds = load_dataset(dataset_name, split=split_name)
    print(f"Dataset loaded. Number of rows = {len(ds)}")

    # 2) Rename the text column if needed. We'll check for 'nl_statement' or 'goal'.
    if "nl_statement" in ds.column_names:
        ds = ds.rename_column("nl_statement", "text")
    elif "goal" in ds.column_names:
        ds = ds.rename_column("goal", "text")
    elif "name" in ds.column_names:
        ds = ds.rename_column("name", "text")
    else:
        # If there's already a "text" column, do nothing, otherwise raise an error
        if "text" not in ds.column_names:
            raise ValueError(
                f"Could not find any suitable column for text among {ds.column_names}."
            )

    print(f"Final columns in dataset: {ds.column_names}")

    # 3) Load the model and tokenizer
    model_name = "AI4M/gemma2-2b-gpt4-more-5-epochs"
    print(f"Loading tokenizer/model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # If no pad token, set it to eos
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4) Tokenize + group text via blocks
    #    We'll pick a block_size of 1024 (or adjust as needed).
    block_size = 1024

    def _map_fn(batch):
        return tokenize_and_group_texts_via_blocks(
            examples=batch,
            tokenizer=tokenizer,
            block_size=block_size,
        )

    # We remove the original columns after mapping, to keep only input_ids/labels
    ds_tokenized = ds.map(
        _map_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc=f"Tokenizing and grouping text into {block_size}-sized blocks ...",
    )

    # 5) Evaluate cross-entropy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2  # adjust as necessary

    print("Computing cross-entropy with blocked data ...")
    avg_ce = compute_dataset_crossentropy_via_blocks(
        tokenized_dataset=ds_tokenized,
        model=model,
        batch_size=batch_size,
        device=device,
    )

    # 6) Print result
    print(f"Cross-entropy (natural log base) = {avg_ce:.4f}")
    # If you want ppl in base e: perplexity = exp(avg_ce)
    perplexity = torch.exp(torch.tensor(avg_ce)).item()
    print(f"Perplexity (exp of cross-entropy) = {perplexity:.4f}")


if __name__ == "__main__":
    main()
