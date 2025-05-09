"""
Functions for preparing evaluation data for Lean4 formalization tasks.
This module handles loading, formatting, and tokenizing Lean4 datasets for evaluation.
"""

import os
from typing import Dict, Tuple, Any, Optional, List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from zip_fit.eval.prompts.eval_lean4_prompt_templates import my_prompt_format
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

def prepare_proofnet_eval_dataset(ds_eval: Dataset, tokenizer: AutoTokenizer, block_size: int = 1024) -> Dataset:
    """
    Prepare ProofNet dataset for evaluation.
    
    Args:
        ds_eval: Raw evaluation dataset
        tokenizer: Tokenizer to use
        block_size: Block size for tokenizing groups of texts
    
    Returns:
        Dataset: Processed dataset ready for evaluation
    """
    # Implementation would need to be customized for ProofNet evaluation
    # This is a placeholder based on the code structure from prepare_train_data.py
    
    # Process and tokenize according to evaluation needs
    ds_eval = ds_eval.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=[col for col in ds_eval.column_names if col != 'text'],
        num_proc=48,
    )
    
    return ds_eval

def prepare_proofnet_tf_eval_dataset(ds_tf_eval: Dataset) -> Dataset:
    """
    Prepare ProofNet dataset for teacher-forced evaluation.
    
    Args:
        ds_tf_eval: Raw evaluation dataset
    
    Returns:
        Dataset: Processed dataset ready for teacher-forced evaluation
    """
    return ds_tf_eval.map(
        lambda batch: {
            'prompt': [format_lean4_prompt(nl_stmt) for nl_stmt in batch['nl_statement']],
            'gold_response': [header for header in batch['header_no_import']]
        },
        batched=True,
        num_proc=48
    )

def prepare_eval_data(config: dict = {}) -> Tuple[List[str], List[str], Dataset]:
    """
    Prepare evaluation data for Lean4 formalization tasks.
    
    Args:
        config: Configuration dictionary with optional parameters
    
    Returns:
        Tuple[List[str], List[str], Dataset]: List of prompts, list of gold headers, and the raw dataset
    """
    # Get dataset parameters from config
    dataset_name: str = config.get('dataset_name', "UDACA/proofnet-v3-lean4")
    split: str = config.get('split', "test")
    
    ds_test = load_dataset(dataset_name, split=split)
    
    # Optionally select a subset based on configuration
    if config.get('max_eval_samples') is not None:
        ds_test = ds_test.select(list(range(min(config['max_eval_samples'], len(ds_test)))))
    
    prompts = [my_prompt_format(row['nl_statement']) for row in ds_test]
    gold_headers = [row['header_no_import'] for row in ds_test]
    
    return prompts, gold_headers, ds_test

def prepare_lean4_eval_datasets(
    tokenizer: AutoTokenizer,
    config: Dict[str, Any] = {}
) -> Tuple[Dataset, Dataset]:
    """
    Prepare Lean4 datasets for evaluation.
    
    Args:
        tokenizer: Tokenizer to use for tokenization
        config: Configuration dictionary with optional parameters
    
    Returns:
        Tuple[Dataset, Dataset]: Evaluation dataset and teacher-forced evaluation dataset
    """
    block_size: int = config.get('block_size', 1024)
    print(f'{block_size=}')
    
    # Get dataset parameters from config
    dataset_name: str = config.get('dataset_name', "UDACA/proofnet-v3-lean4")
    split: str = config.get('split', "test")
    
    # Load ProofNet datasets for evaluation
    raw_eval_dataset = load_dataset(dataset_name, split=split).with_format('torch')
    raw_tf_eval_dataset = load_dataset(dataset_name, split=split).with_format('torch')
    
    # Prepare datasets for evaluation
    eval_dataset = prepare_proofnet_eval_dataset(raw_eval_dataset, tokenizer, block_size)
    tf_eval_dataset = prepare_proofnet_tf_eval_dataset(raw_tf_eval_dataset)
    
    # Print dataset information
    print(f'{len(eval_dataset)=} {eval_dataset.column_names=}')
    print(f'{len(tf_eval_dataset)=} {tf_eval_dataset.column_names=}')
    
    return eval_dataset, tf_eval_dataset
