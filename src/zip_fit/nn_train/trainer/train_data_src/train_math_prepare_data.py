"""
Functions for preparing training data for math language models.
This module handles loading, formatting, and tokenizing math datasets for training.
"""

import os
from typing import Dict, Tuple, Any, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from zip_fit.nn_train.trainer.train_data_src.prepare_tokenization import tokenize_and_group_texts_via_blocks
from zip_fit.nn_train.trainer.prompts.train_math_prompt_templates import format_gsm8k_prompt, get_zipfit_math_train_prompt

def prepare_gsm8k_train_dataset(ds_train: Dataset, tokenizer: AutoTokenizer, block_size: int = 1024) -> Dataset:
    """ Prepare GSM8K dataset for training. """
    # First stage: add "text" field with formatted prompts
    ds_train = ds_train.map(
        lambda batch: {"text": [format_gsm8k_prompt(q, a, fa) for q, a, fa in zip(
            batch["y_grade_school_math_question"], 
            batch["y_grade_school_math_solution"], 
            batch["y_grade_school_math_final_answer"]
        )]},
        batched=True,
        remove_columns=ds_train.column_names,
        num_proc=48,
    )
    
    # Second stage: tokenize and group text into blocks
    ds_train = ds_train.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=ds_train.column_names,
        num_proc=48,
    )
    
    return ds_train

def prepare_math_select_dataset(ds_train: Dataset, tokenizer: AutoTokenizer, block_size: int = 1024) -> Dataset:
    """ Prepare Math Select dataset for training. """
    # No need to process the text, use it directly -- just tokenize and group text into blocks
    ds_train = ds_train.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=[col for col in ds_train.column_names if col != 'text'],
        num_proc=48,
    )
    
    return ds_train

def prepare_math_train_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    split: str, # not defaulted to train because training eval uses same data preparation as training.
    config: Dict[str, Any] = {},
) -> Dataset:
    """ Prepare math dataset for training. """
    block_size: int = config.get('block_size', 1024)
    print(f'{block_size=}')
    
    # Get common parameters
    max_samples: Optional[int] = config.get('max_train_samples', None)
    
    # Load datasets based on name
    if dataset_name == "zipfit/math-select-06062025" or dataset_name.startswith("zipfit/math-select"):
        # Construct split string with max_samples limit if specified
        split_str = f'{split}[:{max_samples}]' if max_samples else split
        
        # Load dataset directly
        raw_train_dataset = load_dataset(dataset_name, split=split_str).with_format('torch')
        
        # Prepare dataset for training
        train_dataset = prepare_math_select_dataset(raw_train_dataset, tokenizer, block_size)

    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    
    # Print dataset information
    print(f'{len(train_dataset)=} {train_dataset.column_names=}')
    return train_dataset 