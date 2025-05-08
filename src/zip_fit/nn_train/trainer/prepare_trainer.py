"""
Functions for preparing training arguments and trainers for language models.
This module handles the configuration of training settings and trainer setup.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from zip_fit.metrics.tfa import TfaCallback

def create_training_args(
    config: Dict[str, Any] = {},
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> TrainingArguments:
    """
    Create training arguments for the Hugging Face Trainer.
    
    Args:
        config: Configuration dictionary with training parameters
        output_dir: Directory to save outputs; if None, creates one based on date
        seed: Random seed for reproducibility
        
    Returns:
        TrainingArguments: Configured training arguments
    """
    # Create output directory if not provided
    if output_dir is None:
        today = config.get('today', datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss'))
        output_dir = Path(f'~/data/zipfit_less_runs/tfa_output_{today}').expanduser()
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training arguments
    training_args = TrainingArguments(
        # max_steps=2, # for debugging
        output_dir=str(output_dir),  # Main output directory.
        do_train=True,
        num_train_epochs=config.get('num_train_epochs', 1),  # Total training epochs.
        do_eval=True,
        eval_on_start=config.get('eval_on_start', True),     # Evaluate before training starts.
        evaluation_strategy=config.get('evaluation_strategy', "steps"),  # Evaluate every few steps.
        eval_steps=config.get('eval_steps', 1),             # Evaluate after every steps.
        logging_steps=config.get('logging_steps', 1),       # Log metrics every steps.
        per_device_train_batch_size=config.get('per_device_train_batch_size', 2),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        save_steps=config.get('save_steps', 25),            # Save a checkpoint every steps.
        save_total_limit=config.get('save_total_limit', 1),  # Keep only the most recent checkpoint.
        save_strategy=config.get('save_strategy', 'steps'),
        bf16=torch.cuda.is_bf16_supported(),               # Use bf16 if supported.
        fp16=not torch.cuda.is_bf16_supported(),            # Otherwise, use fp16.
        optim=config.get('optim', 'paged_adamw_32bit'),
        learning_rate=config.get('learning_rate', 1e-6),
        weight_decay=config.get('weight_decay', 1e-4),
        gradient_checkpointing=config.get('gradient_checkpointing', True),  # careful with hardware
        lr_scheduler_type=config.get('lr_scheduler_type', 'constant_with_warmup'),
        warmup_ratio=config.get('warmup_ratio', 0.05),
        seed=seed,
        data_seed=config.get('data_seed', seed),
        # Additional optional parameters
        **{k: v for k, v in config.items() if k in [
            'push_to_hub', 'hub_model_id', 'remove_unused_columns', 'torch_compile'
        ]}
    )
    
    return training_args

def create_tfa_callback(
    ds_tf_eval: Dataset,
    model_name: str,
    config: Dict[str, Any] = {}
) -> TfaCallback:
    """
    Create a Teacher Forcing Accuracy (TFA) callback for evaluation.
    
    Args:
        ds_tf_eval: Dataset for teacher-forced evaluation
        model_name: Name of the model repository
        config: Configuration dictionary with callback parameters
        
    Returns:
        TfaCallback: Configured TFA callback
    """
    return TfaCallback(
        tfa_dataset=ds_tf_eval,
        repo=model_name,
        n_begin=config.get('n_begin', -1),  # Use full eval set at beginning.
        n_during=config.get('n_during', 4),  # Partial eval during training to save time.
        n_end=config.get('n_end', -1)       # Use full eval set at end.
    )

def create_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tf_eval_dataset: Dataset,
    model_name: str,
    config: Dict[str, Any] = {},
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> Tuple[Trainer, Path]:
    """
    Create a HuggingFace Trainer with all configuration and callbacks.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer for the model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tf_eval_dataset: Teacher-forced evaluation dataset
        model_name: Name of the model (for repository reference)
        config: Configuration dictionary
        output_dir: Output directory (if None, one will be created)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple[Trainer, Path]: Configured trainer and the output directory path
    """
    # Create training arguments
    training_args = create_training_args(config, output_dir, seed)
    
    # Get the actual output directory from the training args
    output_dir = Path(training_args.output_dir)
    
    # Create TFA callback
    tfa_callback = create_tfa_callback(tf_eval_dataset, model_name, config)
    
    # Build the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[tfa_callback],
        tokenizer=tokenizer,  # Ensure tokenizer is saved and pushed.
    )
    
    return trainer, output_dir

