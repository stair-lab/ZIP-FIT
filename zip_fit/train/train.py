# zip_fit/train/train.py

import os
import time
import random
import torch
from typing import Optional, Callable
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import load_dataset, Dataset
import wandb

from tfa import TfaCallback

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



def minimal_trainer(config: dict = {}):
    from transformers import TrainingArguments, Trainer
    from pathlib import Path

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # choose GPU

    # 1) Basic seeding
    seed_everything(42)

    # 2) Load a small model (e.g. GPT-2).
    # model_name = "gpt2"
    model_name = "google/gemma-2-2b"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # 3) Prepare dataset
    def my_prompt_format(nl_stmt: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{nl_stmt}\n"
        )
    def my_prompt_format(nl_stmt: str) -> str:
        # format iddah used for less: https://huggingface.co/datasets/AI4M/less-proofnet-lean4-top1M/viewer/default/train?row=0 
        return f'informal statement {nl_stmt}'
    ds_train = load_dataset("AI4M/less-proofnet-lean4-top1M", split="validation")
    # ds_train = load_dataset("UDACA/proofnet-lean4", split="validation")
    ds_train = ds_train.with_format('torch')  
    ds_train = ds_train.map(
        lambda example: {
            'text': my_prompt_format(example['nl_statement']) 
                     + example['formal_statement'] 
                     + tokenizer.eos_token
        },
        num_proc=24
    )

    def tokenize_function(examples):
        # We create 'input_ids', 'attention_mask' and 'labels' = 'input_ids'
        tokenized = tokenizer(
            examples["text"], 
            padding='max_length', 
            max_length=300, 
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

    ds_eval = ds_eval.map(
        lambda ex: {
            'prompt': my_prompt_format(ex['nl_statement']), 
            'gold_response': ex['formal_statement']
        },
        num_proc=24
    )

    # 4) Minimal training args: run for 1 step, do evaluation at the same step.
    output_dir = Path('~/data/zipfit_less_runs/tfa_output').expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
    #    max_steps=1,                # Only 1 step
        num_train_epochs=1,
        eval_on_start=config.get('eval_on_start', True),
        evaluation_strategy="steps",# Evaluate every 'eval_steps'
        eval_steps=25,               # so we'll evaluate after 1 step
        logging_steps=25,            # log after every step
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        save_strategy="no",
        # **FIX**: disable column pruning
        remove_unused_columns=False,
        save_steps=config.get('save_steps', 50), 
        save_total_limit=1,
        save_strategy=config.get('save_strategy', 'steps'),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        # -- optim
        optim=config.get('optim', 'adamw_torch'),
        # optim=config.get('optim', 'paged_adamw_32bit'),
        learning_rate=config.get('learning_rate', 1e-6),
        weight_decay=config.get('weight_decay', 1e-4),
        # gradient_checkpointing=config.get('gradient_checkpointing', False), # careful might give issues, but not in skampere1
        gradient_checkpointing=config.get('gradient_checkpointing', True), # careful might give issues, but not in skampere1
        # # -- scheduler
        # lr_scheduler_type=config.get('lr_scheduler_type', 'constant'), # this is the hf default btw
        lr_scheduler_type=config.get('lr_scheduler_type', 'constant_with_warmup'), # this is the hf default btw
        # warmup_ratio=config.get('warmup_ratio', 0.0), 
        # # -- seed
        # seed=config.get('seed', 0),
        # data_seed=config.get('data_seed', config.get('seed', 0)),
        torch_compile=True,

    )

    # 5) Attach TfaCallback
    callback = TfaCallback(
        tfa_dataset=ds_eval,
        repo=model_name,
        n_begin=186,
        n_during=185,
        n_end=186
    )

    # 6) Build trainer
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=ds_train,
       eval_dataset=ds_train,  # or ds_eval, whichever you want for HF's standard .evaluate()
       callbacks=[callback]
    )

    # 7) Run training
    trainer.train()