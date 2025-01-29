# tfa.py

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


def teacher_forced_accuracy_tfa(
    prompt: str,
    gold_response: str,
    model: PreTrainedModel,
    repo: str,
    device: str = "cuda"
) -> float:
    """
    Teacher-forced accuracy (token-level) on `gold_response` given a concatenated text = prompt + gold_response.

    Steps:
      1) Combined text = prompt + "\n\n" + gold_response
      2) Tokenize combined text => shape: (1, total_seq_len)
      3) Forward pass => logits shape: (1, total_seq_len, vocab_size)
      4) Identify the token range for the gold_response
      5) Compare the predicted tokens in that range with the reference gold_response tokens
      6) Return fraction matched in [0, 1]

    Notes about BOS/EOS/PAD:
      - Because we do per-example calls (prompt+gold_response) only, no extra padding is needed.
      - We do not forcibly add BOS or EOS here. We skip it to match a "bare-bones" style,
        similar to the updated tfa.py that also ignores explicit BOS/EOS tokens.
      - If the combined text is truncated or too short, we return 0.0 as a fallback.

    """

    # 1) Combine text
    combined_text = prompt + "\n\n" + gold_response

    # 2) Use the tokenizer from the same `repo` to ensure consistency
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # 3) Tokenize entire reference
    enc = tokenizer(combined_text, return_tensors="pt")
    # shape: (1, total_seq_len)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape: (1, total_seq_len, vocab_size)
    preds = torch.argmax(logits, dim=-1)  # shape: (1, total_seq_len)

    # 4) Tokenize the gold_response alone to find how many tokens it has
    gold_response_enc = tokenizer(gold_response, add_special_tokens=False)
    len_gold_response = len(gold_response_enc["input_ids"])

    # Tokenize the prompt alone for length
    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    len_prompt = len(prompt_enc["input_ids"])

    total_seq_len = input_ids.size(1)

    # If the combined text is too short or the gold_response doesn't fit, skip
    if len_prompt + len_gold_response >= total_seq_len:
        return 0.0

    # Teacher forcing alignment:
    #   model's position t attempts to predict token at position t+1
    pred_slice = preds[:, len_prompt : (len_prompt + len_gold_response)]
    label_slice = input_ids[:, (len_prompt + 1) : (len_prompt + 1 + len_gold_response)]

    if pred_slice.size(1) == 0 or label_slice.size(1) == 0:
        return 0.0

    correctness = (pred_slice == label_slice).float()  # shape: (1, number_of_gold_response_tokens)
    acc = correctness.mean().item()
    return acc


def compute_tfa_for_subds(
    sub_ds,
    model: PreTrainedModel,
    repo: str,
    device: str = "cuda",
    debug: bool = False,
) -> float:
    """
    Process an entire subset of data (sub_ds) and compute the average TFA across all examples.

    Parameters:
      sub_ds: The subset of the dataset (like a HuggingFace 'Dataset' slice).
      model:  A language model (transformers PreTrainedModel).
      repo:   The model repo string, used to load the correct tokenizer in teacher_forced_accuracy_tfa.
      device: 'cuda' or 'cpu'.

    Returns:
      float: The average TFA over all examples in sub_ds.
    """
    sum_acc = 0.0
    count = 0

    for i, example in enumerate(sub_ds):
        prompt = example["prompt"]
        gold_response = example["gold_response"]

        acc_i = teacher_forced_accuracy_tfa(
            prompt=prompt,
            gold_response=gold_response,
            model=model,
            repo=repo,
            device=device
        )
        sum_acc += acc_i
        count += 1

        print(f" Example {i}: TFA = {acc_i:.4f}") if debug else None

    return sum_acc / count if count > 0 else 0.0


class TfaCallback(TrainerCallback):
    """
    A callback that performs Teacher-Forced Accuracy (TFA) evaluations at:
      - on_train_begin  => measure TFA on up to `n_begin` samples 
      - on_evaluate     => measure TFA on up to `n_during` samples
      - on_train_end    => measure TFA on up to `n_end` samples (or entire set if n_end == -1)
    """

    def __init__(
        self,
        tfa_dataset: Dataset,
        repo: str,
        n_begin: int = -1,
        n_during: int = 2,
        n_end: int = -1,
        device: str = "cuda"
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
        print(log_dict)
        # print(f"[TfaCallback] on_{label} => TFA = {tfa_score:.4f} on {n_samples} random samples.")
        wandb.log(log_dict)


def main():
    import time

    global_start_time = time.time()  # Start overall timer

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # choose GPU
    seed_everything()

    # 1) Load the ProofNet validation set
    ds = load_dataset("hoskinson-center/proofnet", split="validation")

    # Example of a custom prompt format function
    def my_prompt_format(prompt: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{prompt}\n"
        )
    ds = ds.map(lambda example: {'prompt': my_prompt_format(example['nl_statement']), 'gold_response': example['formal_statement']}, num_proc=24)

    # We'll just do the first N examples for demonstration
    N = 5
    sub_ds = ds.select(range(min(N, len(ds))))

    # 2) Our model list (including all desired models, even if some remain commented)
    model_token_configs = [
        # {
        #     "name": "internlm2-math-plus-1_8b",
        #     "repo": "internlm/internlm2-math-plus-1_8b",
        # },
        {
            "name": "google/gemma-2-2b",
            "repo": "google/gemma-2-2b",
        },
        # {
        #     "name": "Mistral-7B-v0.1",
        #     "repo": "mistralai/Mistral-7B-v0.1",
        # },
        # {
        #     "name": "google/codegemma-2b",
        #     "repo": "google/codegemma-2b",
        # },
        # {
        #     "name": "Meta-Llama-3-8B",
        #     "repo": "meta-llama/Meta-Llama-3-8B",
        # },
        # {
        #     "name": "Meta-Llama-3-8B-Instruct",
        #     "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        # },
        # {
        #     "name": "google/gemma-2-2b-it",
        #     "repo": "google/gemma-2-2b-it",
        # },
        # {
        #     "name": "GPT-2 (small)",
        #     "repo": "gpt2",
        # },
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for config in model_token_configs:
        model_name = config["name"]
        repo = config["repo"]

        print(f"\nEvaluating {model_name} from {repo} on {N} example(s) of ProofNet validation.")

        # Start per-model timer
        model_start_time = time.time()

        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).to(device)
        avg_tfa = compute_tfa_for_subds(
            sub_ds=sub_ds,
            model=model,
            repo=repo,
            device=device
        )

        # End per-model timer
        model_end_time = time.time()
        model_seconds = model_end_time - model_start_time

        print(f" => Average TFA for {model_name} on these {N} example(s) = {avg_tfa:.4f}")
        print(f" => Time to compute TFA for {model_name}: {model_seconds:.2f} seconds.")

        # Test CallBack
        tfacb = TfaCallback(sub_ds, repo, 2, 2, 2)

    # End overall timer
    global_end_time = time.time()
    total_seconds = global_end_time - global_start_time
    print(f"\nDone. Total run time for all models: {total_seconds:.2f} seconds.")


def minimal_tfa_trainer_test():
    """
    A minimal script that demonstrates using the TfaCallback with 
    the Hugging Face Trainer for a tiny "toy" dataset. 
    It runs for 1 training step and triggers the TfaCallback logic 
    at training begin, evaluation, and training end.
    """
    from transformers import TrainingArguments, Trainer

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # choose GPU

    # 1) Basic seeding
    seed_everything(42)

    # 2) Load a small model (e.g. GPT-2).
    model_name = "gpt2"
    model_name = "google/gemma-2-2b"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # 3) Prepare dataset
    def my_prompt_format(prompt: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{prompt}\n"
        )
    ds_train = load_dataset("hoskinson-center/proofnet", split="validation")
    # ds_train = load_dataset("hoskinson-center/proofnet", split="test")
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

    ds_eval = load_dataset("hoskinson-center/proofnet", split="test")
    ds_eval = ds_eval.map(
        lambda ex: {
            'prompt': my_prompt_format(ex['nl_statement']), 
            'gold_response': ex['formal_statement']
        },
        num_proc=24
    )

    # 4) Minimal training args: run for 1 step, do evaluation at the same step.
    training_args = TrainingArguments(
       output_dir="./test-tfa-output",
       do_train=True,
       do_eval=True,
    #    max_steps=1,                # Only 1 step
       num_train_epochs=4,
       evaluation_strategy="steps",# Evaluate every 'eval_steps'
       eval_steps=1,               # so we'll evaluate after 1 step
       logging_steps=1,            # log after every step
       per_device_train_batch_size=4,
       save_strategy="no",
       # **FIX**: disable column pruning
       remove_unused_columns=False
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


if __name__ == "__main__":
    # main()
    minimal_tfa_trainer_test()
