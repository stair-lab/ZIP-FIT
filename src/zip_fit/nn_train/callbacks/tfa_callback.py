"""
Functions for creating and configuring teacher-forced accuracy callbacks.
This module provides a callback to measure teacher forcing accuracy during training.
"""

from typing import Dict, Any
from datasets import Dataset
import random
from tqdm import tqdm
import wandb
from transformers import TrainerCallback, TrainerState, TrainerControl

from zip_fit.metrics.tfa import compute_tfa_for_subds

class TfaCallback(TrainerCallback):
    
    def __init__(
        self,
        tfa_dataset: Dataset,
        repo: str,
        n_begin: int = -1,
        n_during: int = 2,
        n_end: int = -1,
        config: Dict[str, Any] = {}
    ):
        """
        Args:
          tfa_dataset (Dataset):
            The dataset for TFA. Must have 'prompt' & 'gold_response' fields 
            or adapt to your logic.

          repo (str):
            HF repo string for tokenization (the same as your model).

          prompt_format_fn (callable, optional):
            If you need to transform the 'prompt' field. 
            If None, we assume sub_ds already has the final prompt.

          n_begin (int):
            # examples for TFA at train start.
            If 0 or negative => skip.

          n_during (int):
            # examples for TFA at on_evaluate calls.
            If 0 or negative => skip.

          n_end (int):
            # examples for TFA at train end.
            If -1 => entire dataset, else up to n_end random examples. 
            If 0 => skip TFA at train end.

          device (str):
            "cuda" or "cpu" for TFA eval.

        """
        super().__init__()
        self.tfa_dataset = tfa_dataset
        self.repo = repo
        self.n_begin = n_begin
        self.n_during = n_during
        self.n_end = n_end
        self.config = config

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.n_begin == 0:
            return
        # if n_end == -1 => entire dataset
        n = len(self.tfa_dataset) if self.n_begin == -1 else self.n_begin
        self._eval_tfa_and_log(
            n_samples=n,
            label="train_begin",
            state=state,
            **kwargs
        )

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.n_during == 0:
            return
        # if n_during == -1 => entire dataset
        n = len(self.tfa_dataset) if self.n_during == -1 else self.n_during
        self._eval_tfa_and_log(
            n_samples=n,
            label="during_eval",
            state=state,
            **kwargs
        )

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.n_end == 0:
            return
        # if n_end == -1 => entire dataset
        n = len(self.tfa_dataset) if self.n_end == -1 else self.n_end
        self._eval_tfa_and_log(
            n_samples=n,
            label="train_end",
            state=state,
            **kwargs
        )

    def _eval_tfa_and_log(self, n_samples: int, label: str, state: TrainerState, **kwargs):
        """
        A helper function to do the TFA evaluation, random sample up to n_samples from self.tfa_dataset,
        compute TFA, then log/print results with the given label.
        """
        # get model
        model = kwargs["model"]
        current_step = state.global_step
        device = next(model.parameters()).device

        ds_size = len(self.tfa_dataset)
        if n_samples > ds_size:
            n_samples = ds_size
        indices = random.sample(range(ds_size), k=n_samples)
        sub_ds = self.tfa_dataset.select(indices)

        tfa_score = compute_tfa_for_subds(
            sub_ds=sub_ds,
            model=model,
            repo=self.repo,
            device=device
        )
        log_dict = {f"tfa/{label}": tfa_score, "global_step": current_step}
        # print(log_dict)
        tqdm.write(str(log_dict))
        # print(f"[TfaCallback] on_{label} => TFA = {tfa_score:.4f} on {n_samples} random samples.")
        wandb.log(log_dict)
