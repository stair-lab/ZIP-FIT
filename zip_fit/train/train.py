# zip_fit/train/train.py

def seed_everything(seed: int = 42):
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    """
    import random
    import numpy as np
    from transformers import set_seed as hf_set_seed
    import torch

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
    try:
        from vllm import set_seed as vllm_set_seed
        vllm_set_seed(seed)
    except ImportError:
        print("vLLM not installed or vllm set seed has a bug, skipping vLLM seed setting.")


from itertools import chain  # Import chain from the itertools module.

from itertools import chain  # Import chain from the itertools module.

def create_blocks(text_data, tokenizer, block_size):
    """
    Tokenize input text data and split the tokens into fixed-size blocks.

    This function performs the following steps:
      1. Tokenizes each string in the input list `text_data` using the provided `tokenizer`.
      2. Appends the tokenizer's end-of-sequence (EOS) token to the token list of each text.
      3. Concatenates all tokenized texts into a single long list of tokens.
      4. Truncates the token list so that its total length is a multiple of `block_size`.
      5. Splits the truncated token list into contiguous blocks, each of length `block_size`.

    Args:
        text_data (list of str): A list containing the text strings to tokenize.
        tokenizer: An object with:
            - A callable interface that returns a dictionary containing at least the key 'input_ids'
              when a text string is passed in.
            - An attribute `eos_token_id` which provides the token ID for the end-of-sequence.
        block_size (int): The desired number of tokens per block.

    Returns:
        list of list of int: A list where each element is a block (a list) of token IDs, each of length `block_size`.
    """

    # Retrieve the end-of-sequence (EOS) token ID from the tokenizer.
    eos_token_id = tokenizer.eos_token_id

    # For each text in the input list:
    #   - Tokenize the text to get a dictionary; extract the token IDs from 'input_ids'.
    #   - Append the EOS token ID to the token list.
    # The list comprehension creates a list of lists of tokens (one per text).
    #
    # The asterisk operator (*) in front of the list comprehension unpacks the list of lists,
    # meaning that each inner list is passed as a separate argument to the chain function
    # eg chain(*[[1,2],[3,4]]) -> chain([1,2],[3,4]). 
    # Then the chain function, imported from itertools, takes multiple iterables as arguments and
    # returns a single iterator that yields elements from the first iterable, then the second,
    # and so on, effectively flattening the list of lists into one long list of tokens
    # effectively chain([1,2],[3,4]) ~ [1,2,3,4] via a generator
    # For example, chain([1, 2], [3, 4]) (chain of tok seqs) yields: 1, 2, 3, 4 (return each tok as if it was one long tok seq).
    concatenated_tokens = list(
        chain(*[
            tokenizer(text)['input_ids'] + [eos_token_id]  # Tokenize text and append EOS token.
            for text in text_data
        ])
    )

    # Compute the total number of tokens in the concatenated list.
    total_length = len(concatenated_tokens)

    # Adjust the total length to be an exact multiple of block_size by discarding any extra tokens
    # at the end that would not fill a complete block.
    # This is achieved by performing integer division (//) of total_length by block_size,
    # which computes the number of complete blocks that can be formed.
    # Multiplying that number by block_size yields the total number of tokens that exactly fit into those blocks,
    # effectively discarding any remaining tokens that would not complete a full block.
    total_length = (total_length // block_size) * block_size

    # Split the concatenated token list into blocks of size block_size.
    # The list comprehension iterates over the token list in steps of block_size,
    # slicing out a block each time.
    # Because total_length has been truncated to be an exact multiple of block_size,
    # each slice taken from index i to i + block_size will contain exactly block_size tokens.
    # This ensures the entire token list is partitioned into equally sized blocks without any leftovers.
    all_tokens = [
        concatenated_tokens[i: i + block_size]
        for i in range(0, total_length, block_size)
    ]

    # Return the list of token blocks.
    return all_tokens

def main_train(config: dict = {}):
    from transformers import TrainingArguments, Trainer
    from pathlib import Path
    import datetime
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
    # def my_prompt_format(nl_stmt: str) -> str:
    #     return (
    #         "Translate the natural language version of the mathematical statement "
    #         f"to a formal Lean version:\n{nl_stmt}\n"
    #     )
    def my_prompt_format(nl_stmt: str) -> str:
        # format iddah used for less: https://huggingface.co/datasets/AI4M/less-proofnet-lean4-top1M/viewer/default/train?row=0 
        return f'informal statement {nl_stmt}'
    ds_train = load_dataset("AI4M/less-proofnet-lean4-top1M", split="validation")
    # ds_train = load_dataset("UDACA/proofnet-lean4", split="validation")
    ds_train = ds_train.with_format('torch')  
    ds_train = ds_train.map(
        lambda example: {
            'text': my_prompt_format(example['text']) 
                     + tokenizer.eos_token
        },
        num_proc=24
    )
    def tokenize_function(examples):
        # We create 'input_ids', 'attention_mask' and 'labels' = 'input_ids'
        tokenized = tokenizer(
            examples["text"], 
            padding='max_length', 
            max_length=512, 
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
            'prompt': my_prompt_format(ex['informal_prefix']), 
            'gold_response': ex['formal_statement']
        },
        num_proc=24
    )

    # 4) Minimal training args: run for 1 step, do evaluation at the same step.
    today: str = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss')
    output_dir = Path(f'~/data/zipfit_less_runs/tfa_output_{today}').expanduser()
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
        warmup_ratio=config.get('warmup_ratio', 0.05), 
        # # -- seed
        seed=config.get('seed', 42),
        data_seed=config.get('data_seed', config.get('seed', 42)),
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

def get_current_tmux_session_number() -> str:
    import os
    # Executes "tmux display-message -p '#S'" to retrieve the current tmux session name/number,
    # reads the output from the command, strips any surrounding whitespace, and returns it.
    return os.popen("tmux display-message -p '#S'").read().strip()

def _main(**kwargs):
    from datetime import datetime
    from socket import gethostname
    import wandb
    today = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss') # eg '2024_m01_d22_t13h_00m_30s'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(kwargs.get('CUDA_VISIBLE_DEVICES', '7'))
    tmux_sess_num = None
    kwargs = kwargs | {'today': today, 'tmux_sess_num': tmux_sess_num, 'hostname': gethostname()}
    run_name = f'{kwargs}' 
    run = wandb.init(mode=kwargs.get('mode', 'online'), project="zip-fit-pass-at-k-af", name=run_name, save_code=True, config=kwargs)
    # run = wandb.init(mode=kwargs.get('mode', 'dryrun'), project="zip-fit-pass-at-k-af", name=run_name, save_code=True, config=kwargs)
    wandb.save(__file__) # save current code now, don't wait to wandb.finish, also useful: wandb.save("*.py") # upload all .py files in current directory
    print(f'Kwargs to run:\n{kwargs}')
    # main(kwargs)
    main_train(kwargs)
    run.alert(title="Run Completed", text=f"Run finished, run url: {run.get_url()}")
    print(f'{run.get_url()=}')
    wandb.finish()

if __name__ == "__main__":
    import fire
    import time
    start_time = time.time()
    fire.Fire(_main)
    print(f"\aTime taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
