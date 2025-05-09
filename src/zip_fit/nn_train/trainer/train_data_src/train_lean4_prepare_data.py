"""
Functions for preparing training data for Lean4 formalization tasks.
This module handles loading, formatting, and tokenizing Lean4 datasets for training.
"""

import os
from typing import Dict, Tuple, Any, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from zip_fit.nn_train.trainer.train_data_src.prepare_tokenization import tokenize_and_group_texts_via_blocks

# -------------------------------------------------------------------------
# Prompt formatting functions
# -------------------------------------------------------------------------

def format_lean4_prompt(nl_stmt: str) -> str:
    """
    Format a prompt for Lean4 formalization tasks.
    """
    return f'informal statement {nl_stmt}'

# -------------------------------------------------------------------------
# Dataset preparation functions
# -------------------------------------------------------------------------

def prepare_proofnet_train_dataset(ds_train: Dataset, tokenizer: AutoTokenizer, block_size: int = 1024) -> Dataset:
    """
    Prepare ProofNet dataset for training.
    
    Args:
        ds_train: Raw training dataset
        tokenizer: Tokenizer to use
        block_size: Block size for tokenizing groups of texts
    
    Returns:
        Dataset: Processed dataset ready for training
    """
    # Implementation would need to be customized for ProofNet training
    # For now, just as a placeholder based on the prepare_train_data.py structure
    
    # Add a "text" field with formatted prompts if needed
    # This would depend on the specific format needed for Lean4 training
    
    # Example (needs to be adapted for actual Lean4 training):
    ds_train = ds_train.map(
        lambda batch: {"text": [format_lean4_prompt(nl_stmt) + " " + header 
                               for nl_stmt, header in zip(batch["nl_statement"], batch["header_no_import"])]},
        batched=True,
        remove_columns=ds_train.column_names,
        num_proc=48,
    )
    
    # Tokenize and group text into blocks
    ds_train = ds_train.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=ds_train.column_names,
        num_proc=48,
    )
    
    return ds_train

# -------------------------------------------------------------------------
# Main data preparation function
# -------------------------------------------------------------------------

def prepare_lean4_train_dataset(
    tokenizer: AutoTokenizer,
    config: Dict[str, Any] = {}
) -> Dataset:
    """
    Prepare Lean4 dataset for training.
    
    Args:
        tokenizer: Tokenizer to use for tokenization
        config: Configuration dictionary with optional parameters
    
    Returns:
        Dataset: Training dataset
    """
    block_size: int = config.get('block_size', 1024)
    print(f'{block_size=}')
    
    # Get dataset parameters from config
    dataset_name: str = config.get('dataset_name', "UDACA/proofnet-v3-lean4")
    split: str = config.get('split', "train")
    
    # Load ProofNet dataset for training
    raw_train_dataset = load_dataset(dataset_name, split=split).with_format('torch')
    
    # Apply max samples limit if specified
    max_train_samples = config.get('max_train_samples')
    if max_train_samples is not None:
        raw_train_dataset = raw_train_dataset.select(range(min(len(raw_train_dataset), max_train_samples)))
    
    # Prepare dataset for training
    train_dataset = prepare_proofnet_train_dataset(raw_train_dataset, tokenizer, block_size)
    
    # Print dataset information
    print(f'{len(train_dataset)=} {train_dataset.column_names=}')
    
    return train_dataset 