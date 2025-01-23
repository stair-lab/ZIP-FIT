# tfa_callback.py

import random
import torch
from typing import List, Optional
import wandb

from tfa import teacher_forced_accuracy_tfa 
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import SFTTrainer

class TfaCallback(TrainerCallback):
    """
    A callback that computes teacher-forced accuracy (TFA) on a proofnet *test* set
    at different points in training:
      - on_train_begin: TFA on 5 random examples (model is untrained).
      - on_evaluate (or on_save): TFA on 2 random examples (progress check).
      - on_train_end: TFA on the *entire* test set or large subset (final improvement).

    The code calls your teacher_forced_accuracy_tfa(...) for each example, averages, then logs to wandb.
    """

    def __init__(
        self,
        test_dataset: Dataset, 
        model_repo: str,
        device: str = "cuda",
        prompt_format_fn=None,
    ):
        """
        Args:
          test_dataset: A huggingface `Dataset` (split='test' or 'validation') from proofnet.
          model_repo:   The huggingface repo string (eg. "google/gemma-2-2b") so we can load the same tokenizer.
          device:       'cuda' or 'cpu'.
          prompt_format_fn: optional function that transforms an 'nl_statement' into a 'prompt'.
        """
        self.test_dataset = test_dataset
        self.model_repo = model_repo
        self.device = device
        self.prompt_format_fn = prompt_format_fn

        self.n_train_begin: int = len(self.test_dataset),

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the start of training: Evaluate TFA on 5 random examples (untrained model).
        """
        trainer = kwargs["trainer"]  # SFTTrainer
        model = trainer.model
        
        # pick 5 random examples from test_dataset
        sample_size = min(5, len(self.test_dataset))
        indices = random.sample(range(len(self.test_dataset)), sample_size)
        subset = [self.test_dataset[i] for i in indices]

        tfa = self._compute_tfa_on_subset(subset, model)
        print(f"[TFA Callback] On train begin - TFA (5 random examples) = {tfa:.4f}")
        wandb.log({"tfa/train_begin_5examples": tfa, "train_step": state.global_step})

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at each 'evaluate' event: Evaluate TFA on 2 random examples (progress).
        """
        trainer = kwargs["trainer"]
        model = trainer.model
        
        sample_size = min(2, len(self.test_dataset))
        indices = random.sample(range(len(self.test_dataset)), sample_size)
        subset = [self.test_dataset[i] for i in indices]

        tfa = self._compute_tfa_on_subset(subset, model)
        print(f"[TFA Callback] On evaluate - TFA (2 random examples) = {tfa:.4f} at global_step={state.global_step}")
        wandb.log({"tfa/during_training_2examples": tfa, "train_step": state.global_step})

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of training: Evaluate TFA on the entire test set (or large subset).
        """
        trainer = kwargs["trainer"]
        model = trainer.model
        
        # Let's do the entire test set or large subset:
        # for demonstration, do the entire test dataset.
        subset = [self.test_dataset[i] for i in range(len(self.test_dataset))]

        tfa = self._compute_tfa_on_subset(subset, model)
        print(f"[TFA Callback] On train end - TFA (ENTIRE test set) = {tfa:.4f}")
        wandb.log({"tfa/train_end_full_test": tfa, "train_step": state.global_step})

    def _compute_tfa_on_subset(self, subset, model) -> float:
        """
        Compute TFA over `subset` by calling `teacher_forced_accuracy_tfa` for each example,
        then averaging.
        """
        sum_acc = 0.0
        count = 0

        for ex in subset:
            nl_statement = ex["nl_statement"]
            formal_statement = ex["formal_statement"]

            if self.prompt_format_fn is not None:
                prompt = self.prompt_format_fn(nl_statement)
            else:
                prompt = f"Translate the statement to Lean:\n{nl_statement}\n"

            acc_i = teacher_forced_accuracy_tfa(
                prompt=prompt,
                response=formal_statement,
                model=model,
                repo=self.model_repo,
                device=self.device
            )
            sum_acc += acc_i
            count += 1

        return sum_acc / count if count>0 else 0.0
