"""
Functions for preparing training data for language models.
This module handles loading, formatting, and tokenizing datasets for training.
"""

import os
from typing import Dict, List, Tuple, Any, Callable, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from zip_fit.nn_train.nn_train_utils import tokenize_and_group_texts_via_blocks
from zip_fit.nn_train.prompts.train_math_prompt_templates import get_zipfit_math_train_prompt

# -------------------------------------------------------------------------
# Prompt formatting functions
# -------------------------------------------------------------------------

def format_lean4_prompt(nl_stmt: str) -> str:
    """
    Format a prompt for Lean4 formalization tasks.
    """
    return f'informal statement {nl_stmt}'

def format_gsm8k_prompt(question: str, answer: str, final_answer: str) -> str:
    """
    Format a prompt for GSM8K math problems.
    Args:
        question: The math question
        answer: The detailed solution steps
        final_answer: The final answer to the problem
    """
    return f'question: {question}\nanswer: {answer}\n### {final_answer}'

def format_gsm8k_eval_prompt(question: str) -> str:
    """
    Format a prompt for GSM8K math problem evaluation.
    """
    return f'question: {question}\nanswer: '

def format_gsm8k_text(question: str, answer: str) -> str:
    """
    Format GSM8K question and answer for evaluation.
    """
    return f'question: {question}\nanswer: {answer}'

def format_putnam_train_prompt(problem: str, solution: str) -> str:
    """
    Format Putnam problem and solution for training.
    """
    return get_zipfit_math_train_prompt(problem=problem, solution=solution)

def format_putnam_eval_prompt(problem: str) -> str:
    """
    Format Putnam problem for evaluation.
    """
    return f'Problem:\n{problem}\n\nSolution:'

# -------------------------------------------------------------------------
# Dataset loading and preparation functions
# -------------------------------------------------------------------------

def load_gsm8k_dataset() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load GSM8K dataset for training and evaluation.
    
    Returns:
        Tuple[Dataset, Dataset, Dataset]: Training dataset, evaluation dataset, and teacher-forced evaluation dataset
    """
    ds_eval = load_dataset("openai/gsm8k", 'main', split="test").with_format('torch')
    ds_tf_eval = load_dataset("openai/gsm8k", 'main', split="test").with_format('torch')
    
    return None, ds_eval, ds_tf_eval  # Training dataset is None by default

def load_synthetic_dataset(max_samples: Optional[int] = None) -> Dataset:
    """
    Load synthetic dataset from a JSON file.
    
    Args:
        max_samples: Maximum number of samples to load. If None, load all.
    
    Returns:
        Dataset: The loaded dataset
    """
    split_str = f'train[:{max_samples}]' if max_samples else 'train'
    data_path = os.path.expanduser("~/data/synthetic_data/uncompiled_dspy/syndata_18612_2025_m02_d05_t18h_38m_11s.json")
    return load_dataset("json", split=split_str, data_files=data_path)

def load_proofnet_dataset(split: str = "validation") -> Dataset:
    """
    Load ProofNet dataset for Lean4 formalization tasks.
    
    Args:
        split: Dataset split to load ("train", "validation", or "test")
    
    Returns:
        Dataset: The loaded dataset
    """
    return load_dataset("UDACA/proofnet-v3-lean4", split=split).with_format('torch')

def load_math_select_dataset(max_samples: Optional[int] = None) -> Dataset:
    """
    Load ZipFit math-select dataset.
    
    Args:
        max_samples: Maximum number of samples to load. If None, load all.
    
    Returns:
        Dataset: The loaded dataset
    """
    split_str = f'train[:{max_samples}]' if max_samples else 'train'
    return load_dataset("zipfit/math-select-06062025", split=split_str).with_format('torch')

def load_putnam_dataset(split: str = "test") -> Dataset:
    """
    Load Putnam-AXIOM dataset for evaluation.
    
    Args:
        split: Dataset split to load ("validation" or "test")
    
    Returns:
        Dataset: The loaded dataset
    """
    return load_dataset("zipfit/Putnam-AXIOM-for-zip-fit-splits", split=split).with_format('torch')

def prepare_gsm8k_train_dataset(ds_train: Dataset, tokenizer: AutoTokenizer, block_size: int = 1024) -> Dataset:
    """
    Prepare GSM8K dataset for training.
    
    Args:
        ds_train: Raw training dataset
        tokenizer: Tokenizer to use
        block_size: Block size for tokenizing groups of texts
    
    Returns:
        Dataset: Processed dataset ready for training
    """
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
    """
    Prepare Math Select dataset for training.
    
    Args:
        ds_train: Raw training dataset
        tokenizer: Tokenizer to use
        block_size: Block size for tokenizing groups of texts
    
    Returns:
        Dataset: Processed dataset ready for training
    """
    # No need to process the text, use it directly
    # Just tokenize and group text into blocks
    ds_train = ds_train.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=[col for col in ds_train.column_names if col != 'text'],
        num_proc=48,
    )
    
    return ds_train

def prepare_putnam_train_dataset(ds_train: Dataset, tokenizer: AutoTokenizer, block_size: int = 1024) -> Dataset:
    """
    Prepare Putnam dataset for training.
    
    Args:
        ds_train: Raw training dataset
        tokenizer: Tokenizer to use
        block_size: Block size for tokenizing groups of texts
    
    Returns:
        Dataset: Processed dataset ready for training
    """
    # First stage: add "text" field with formatted prompts
    ds_train = ds_train.map(
        lambda batch: {"text": [format_putnam_train_prompt(p, s) for p, s in zip(
            batch["problem"], 
            batch["solution"]
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

def prepare_putnam_eval_dataset(ds_eval: Dataset, tokenizer: AutoTokenizer, block_size: int = 1024) -> Dataset:
    """
    Prepare Putnam dataset for evaluation.
    
    Args:
        ds_eval: Raw evaluation dataset
        tokenizer: Tokenizer to use
        block_size: Block size for tokenizing groups of texts
    
    Returns:
        Dataset: Processed dataset ready for evaluation
    """
    # First stage: add "text" field with formatted prompts
    ds_eval = ds_eval.map(
        lambda batch: {"text": [format_putnam_train_prompt(p, s) for p, s in zip(
            batch["problem"], 
            batch["solution"]
        )]},
        batched=True,
        remove_columns=ds_eval.column_names,
        num_proc=48,
    )
    
    # Second stage: tokenize and group text into blocks
    ds_eval = ds_eval.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=ds_eval.column_names,
        num_proc=48,
    )
    
    return ds_eval

def prepare_putnam_tf_eval_dataset(ds_tf_eval: Dataset) -> Dataset:
    """
    Prepare Putnam dataset for teacher-forced evaluation.
    
    Args:
        ds_tf_eval: Raw evaluation dataset
    
    Returns:
        Dataset: Processed dataset ready for teacher-forced evaluation
    """
    return ds_tf_eval.map(
        lambda batch: {
            'prompt': [format_putnam_eval_prompt(p) for p in batch['problem']],
            'gold_response': [solution for solution in batch['solution']]
        },
        batched=True,
        num_proc=48
    )

def prepare_gsm8k_eval_dataset(ds_eval: Dataset, tokenizer: AutoTokenizer, block_size: int = 1024) -> Dataset:
    """
    Prepare GSM8K dataset for evaluation.
    
    Args:
        ds_eval: Raw evaluation dataset
        tokenizer: Tokenizer to use
        block_size: Block size for tokenizing groups of texts
    
    Returns:
        Dataset: Processed dataset ready for evaluation
    """
    # First stage: add "text" field with formatted prompts
    ds_eval = ds_eval.map(
        lambda batch: {"text": [format_gsm8k_text(q, a) for q, a in zip(
            batch["question"], batch["answer"]
        )]},
        batched=True,
        remove_columns=ds_eval.column_names,
        num_proc=48,
    )
    
    # Second stage: tokenize and group text into blocks
    ds_eval = ds_eval.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=ds_eval.column_names,
        num_proc=48,
    )
    
    return ds_eval

def prepare_tf_eval_dataset(ds_tf_eval: Dataset) -> Dataset:
    """
    Prepare dataset for teacher-forced evaluation.
    
    Args:
        ds_tf_eval: Raw evaluation dataset
    
    Returns:
        Dataset: Processed dataset ready for teacher-forced evaluation
    """
    return ds_tf_eval.map(
        lambda batch: {
            'prompt': [format_gsm8k_eval_prompt(q) for q in batch['question']],
            'gold_response': [answer for answer in batch['answer']]
        },
        batched=True,
        num_proc=48
    )

# -------------------------------------------------------------------------
# Main data preparation function
# -------------------------------------------------------------------------

def prepare_datasets(
    tokenizer: AutoTokenizer,
    config: Dict[str, Any] = {},
    dataset_type: str = "zipfit/math-select-06062025"
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare datasets for training and evaluation.
    
    Args:
        tokenizer: Tokenizer to use for tokenization
        config: Configuration dictionary with optional parameters
        dataset_type: Type of dataset to use ("gsm8k_synthetic", "proofnet", "zipfit/math-select-06062025", etc.)
    
    Returns:
        Tuple[Dataset, Dataset, Dataset]: Training dataset, evaluation dataset, and teacher-forced evaluation dataset
    """
    block_size: int = config.get('block_size', 1024)
    print(f'{block_size=}')
    
    # Load datasets based on type
    if dataset_type == "gsm8k_synthetic":
        # Load synthetic dataset for training
        raw_train_dataset = load_synthetic_dataset(max_samples=config.get('max_train_samples', None))
        
        # Load GSM8K for evaluation
        _, raw_eval_dataset, raw_tf_eval_dataset = load_gsm8k_dataset()
        
        # Prepare datasets for training and evaluation
        train_dataset = prepare_gsm8k_train_dataset(raw_train_dataset, tokenizer, block_size)
        eval_dataset = prepare_gsm8k_eval_dataset(raw_eval_dataset, tokenizer, block_size)
        tf_eval_dataset = prepare_tf_eval_dataset(raw_tf_eval_dataset)
        
    elif dataset_type == "proofnet":
        # Load ProofNet datasets
        raw_train_dataset = load_proofnet_dataset(split="train")
        raw_eval_dataset = load_proofnet_dataset(split="test")
        raw_tf_eval_dataset = load_proofnet_dataset(split="test")
        
        # Prepare datasets for training and evaluation (would need custom functions for ProofNet)
        # For now, using GSM8K preparation as a template
        train_dataset = prepare_gsm8k_train_dataset(raw_train_dataset, tokenizer, block_size)
        eval_dataset = prepare_gsm8k_eval_dataset(raw_eval_dataset, tokenizer, block_size)
        tf_eval_dataset = prepare_tf_eval_dataset(raw_tf_eval_dataset)
    
    elif dataset_type == "zipfit/math-select-06062025":
        # Load Math Select dataset for training
        raw_train_dataset = load_math_select_dataset(max_samples=config.get('max_train_samples', None))
        
        # Load Putnam dataset for evaluation
        raw_eval_dataset = load_putnam_dataset(split="test")
        raw_tf_eval_dataset = load_putnam_dataset(split="test")
        
        # Prepare datasets for training and evaluation
        train_dataset = prepare_math_select_dataset(raw_train_dataset, tokenizer, block_size)
        eval_dataset = prepare_putnam_eval_dataset(raw_eval_dataset, tokenizer, block_size)
        tf_eval_dataset = prepare_putnam_tf_eval_dataset(raw_tf_eval_dataset)
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # Print dataset information
    print(f'{len(train_dataset)=} {train_dataset.column_names=}')
    print(f'{len(eval_dataset)=} {eval_dataset.column_names=}')
    print(f'{len(tf_eval_dataset)=} {tf_eval_dataset.column_names=}')
    
    return train_dataset, eval_dataset, tf_eval_dataset
