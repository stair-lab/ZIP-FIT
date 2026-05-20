"""
Functions for preparing training arguments specific to supervised fine-tuning (SFT).
This module extends base training arguments with SFT-specific configurations.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from transformers import TrainingArguments

from zip_fit.nn_train.trainer.prepare_trainer_args import create_training_args

def create_sft_training_args(
    config: Dict[str, Any] = {},
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> TrainingArguments:
    """
    Create training arguments for Supervised Fine-Tuning.
    This extends the base training arguments with SFT-specific settings.
    """
    raise NotImplementedError("SFT training arguments not yet implemented")