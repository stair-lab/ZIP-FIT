"""
Functions for preparing trainers for language models.
This module handles the configuration of trainer setup.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import torch
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from datasets import Dataset

from zip_fit.nn_train.trainer.prepare_trainer_args import create_training_args
from zip_fit.nn_train.callbacks.tfa_callback import TfaCallback

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
    """
    # Create training arguments
    training_args: TrainingArguments = create_training_args(config, output_dir, seed)
    
    # Get the actual output directory from the training args
    output_dir: Path = Path(training_args.output_dir)
    
    # Create TFA callback only if tf_eval_dataset exists
    callbacks: List[TrainerCallback] = []
    if tf_eval_dataset is not None:
        tfa_callback: TfaCallback = TfaCallback(
            tf_eval_dataset, 
            model_name,
            n_begin=config.get('n_begin', -1), # -1 means all examples
            n_during=config.get('n_during', 2), # 2 means 2 examples
            n_end=config.get('n_end', -1), # -1 means all examples
            config=config
        )
        callbacks.append(tfa_callback)
    
    # Build the Trainer
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,  # This can safely be an empty list
        tokenizer=tokenizer,  # Ensure tokenizer is saved and pushed.
    )
    
    return trainer, output_dir

