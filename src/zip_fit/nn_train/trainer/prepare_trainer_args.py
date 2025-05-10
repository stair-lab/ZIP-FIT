"""
Functions for preparing training arguments for language models.
This module handles the configuration of training settings.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import TrainingArguments

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
        max_steps=config.get('max_steps', -1),  # -1 means train for full epochs, positive for limited steps
        output_dir=str(output_dir),  # Main output directory.
        do_train=True,
        num_train_epochs=config.get('num_train_epochs', 1),  # Total training epochs.
        do_eval=True,
        eval_on_start=config.get('eval_on_start', True),     # Evaluate before training starts.
        eval_strategy=config.get('eval_strategy', "steps"),  # Evaluate every few steps.
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
