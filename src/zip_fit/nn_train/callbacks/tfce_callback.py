from random import random
from datasets import Dataset
from transformers import TrainerCallback, TrainerState, TrainerControl
import wandb

from zip_fit.metrics.tfce import compute_tfce_for_subds

class TfceCallback(TrainerCallback):
    """
    A callback that performs Teacher-Forced Cross-Entropy (TFCE) evaluations at:
      - on_train_begin => up to n_begin samples
      - on_evaluate    => up to n_during samples
      - on_train_end   => up to n_end samples (or entire set if n_end == -1)
    
    Very similar to TfaCallback but uses cross-entropy (tfce_loss_full_gold_ref).
    """

    def __init__(
        self,
        tfce_dataset: Dataset,
        repo: str,
        n_begin: int = -1,
        n_during: int = 2,
        n_end: int = -1,
        reduction: str = "mean"
    ):
        """
        Args:
          tfce_dataset (Dataset):
            A dataset with 'prompt' & 'gold_response' columns for computing TFCE.

          repo (str):
            HF repo string for tokenization.

          n_begin (int):
            # examples for TFCE at train start. If 0 => skip. If -1 => entire dataset.

          n_during (int):
            # examples for TFCE at on_evaluate. If 0 => skip. If -1 => entire dataset.

          n_end (int):
            # examples for TFCE at train end. If 0 => skip. If -1 => entire dataset.

          device (str):
            "cuda" or "cpu".

          reduction (str):
            "mean" or "sum" for cross-entropy reduction.
        """
        super().__init__()
        self.tfce_dataset = tfce_dataset
        self.repo = repo
        self.n_begin = n_begin
        self.n_during = n_during
        self.n_end = n_end
        self.reduction = reduction

    def on_train_begin(self, args: TrainerState, state: TrainerState, control: TrainerControl, **kwargs):
        if self.n_begin == 0:
            return

        # if n_begin == -1 => entire dataset
        n = len(self.tfce_dataset) if self.n_begin == -1 else self.n_begin
        self._eval_tfce_and_log(n_samples=n, label="train_begin", state=state, **kwargs)

    def on_evaluate(self, args: TrainerState, state: TrainerState, control: TrainerControl, **kwargs):
        if self.n_during == 0:
            return

        # if n_during == -1 => entire dataset
        n = len(self.tfce_dataset) if self.n_during == -1 else self.n_during
        self._eval_tfce_and_log(n_samples=n, label="during_eval", state=state, **kwargs)

    def on_train_end(self, args: TrainerState, state: TrainerState, control: TrainerControl, **kwargs):
        if self.n_end == 0:
            return

        # if n_end == -1 => entire dataset
        n = len(self.tfce_dataset) if self.n_end == -1 else self.n_end
        self._eval_tfce_and_log(n_samples=n, label="train_end", state=state, **kwargs)

    def _eval_tfce_and_log(self, n_samples: int, label: str, state: TrainerState, **kwargs):
        """
        Helper that randomly samples up to n_samples from self.tfce_dataset,
        calls compute_tfce_for_subds, logs the result.
        """
        model = kwargs["model"]
        current_step = state.global_step
        device = next(model.parameters()).device

        ds_size = len(self.tfce_dataset)
        if ds_size == 0:
            return
        if n_samples > ds_size:
            n_samples = ds_size

        indices = random.sample(range(ds_size), k=n_samples)
        sub_ds = self.tfce_dataset.select(indices)

        tfce_val = compute_tfce_for_subds(
            sub_ds=sub_ds,
            model=model,
            repo=self.repo,
            device=device,
            reduction=self.reduction
        )
        log_dict = {f"tfce/{label}": tfce_val, "global_step": current_step}
        wandb(log_dict)
        wandb.log(log_dict)