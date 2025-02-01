# zip_fit/train/train.py

from typing import List, Optional, Dict

from transformers import (
    AutoTokenizer,
)

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


def tokenize_and_group_texts_via_blocks(
    examples: Dict[str, List[str]],  # since Batched=True gives a list of strings, note: a block size will be 1 sequences, a concat of tokenized rows from the data set! 
    tokenizer: AutoTokenizer,
    block_size: int,
) -> Dict[str, List[List[int]]]:
    """
    Tokenizes a batch of raw text examples and groups the tokens into fixed-size blocks.

    This function is designed for use with Hugging Face datasets (with batched=True). It processes
    a batch of examples (where each example is a string under the 'text' key) by performing the following steps:

      1. **Retrieve Special Token IDs:**  
         It retrieves the beginning-of-sequence (BOS) token ID and the end-of-sequence (EOS) token ID from the tokenizer.
         - The BOS token ID is obtained from `tokenizer.bos_token_id` (if available).
         - The EOS token ID is obtained from `tokenizer.eos_token_id`.

      2. **Per-Text Tokenization with Special Tokens:**  
         For each text in the input list:
           - It prepends the BOS token (if available).
           - Tokenizes the text using the tokenizer (without adding any special tokens automatically).
           - Appends the EOS token to the token list.
         This is done individually for each text so that we can explicitly control the addition of special tokens.

      3. **Concatenation:**  
         All token lists are concatenated into one long list of tokens. This is achieved using the `chain` function
         to flatten the list of lists.

      4. **Truncation:**  
         The total token list is truncated so that its length is an exact multiple of `block_size`.  
         This is done by discarding any tokens at the end that would not complete a full block.

      5. **Blocking:**  
         The truncated token list is split into contiguous blocks, each of length `block_size`.

      6. **Label Creation:**  
         For language modeling tasks, the function creates a 'labels' field that is an exact copy of the 'input_ids'
         blocks.

    **Note on the attention mask field:**  
    The returned dictionary does not include an `attention_mask` field. Including an attention mask with a value of `None` 
    may cause issues with the Hugging Face Trainer, which expects either a valid tensor or for the key to be omitted entirely.

    Args:
        examples (Dict[str, List[str]]):
            A dictionary representing a batch of examples from the dataset. It must contain a key 'text'
            whose value is a list of strings (one per example).
        tokenizer (Any):
            A tokenizer instance (e.g., from Hugging Face's transformers) that can process a single text string.
            The tokenizer should return a dictionary containing at least the key 'input_ids' when called with a text,
            and it must have attributes `bos_token_id` (which may be None) and `eos_token_id`.
        block_size (int):
            The desired number of tokens per block (for example, 1024 or 4096).

    Returns:
        Dict[str, List[List[int]]]:
            A dictionary where:
              - 'input_ids' maps to a list of token blocks, each block being a list of token IDs of length `block_size`.
              - 'labels' maps to a copy of the 'input_ids' blocks.
            
            **Note:** No `attention_mask` is included in the returned dictionary.
    """
    from itertools import chain
    # -------------------------------------------------------------------------
    # Step 1: Retrieve Special Token IDs.
    # Retrieve the end-of-sequence (EOS) token ID from the tokenizer.
    eos_token_id: int = tokenizer.eos_token_id
    # Retrieve the beginning-of-sequence (BOS) token ID from the tokenizer.
    # If not used, bos_token_id will be None.
    bos_token_id: Optional[int] = tokenizer.bos_token_id

    # -------------------------------------------------------------------------
    # Step 2: Per-Text Tokenization with Special Tokens.
    # For each text in the input list, perform the following:
    #   - Prepend the BOS token if it exists.
    #   - Tokenize the text to get a dictionary; extract the token IDs from 'input_ids'.
    #   - Append the EOS token to the token list.
    # The list comprehension processes each text individually.
    token_lists: List[List[int]] = [
        # If a BOS token exists, prepend it; otherwise, use an empty list.
        ([bos_token_id] if bos_token_id is not None else []) +
        tokenizer(text)['input_ids'] + [eos_token_id]  # Tokenize text and append EOS token.
        for text in examples["text"]
    ]

    # -------------------------------------------------------------------------
    # Step 3: Concatenate tokenized outputs across the batch.
    # Flatten the list of token lists into a single long list using chain.
    concatenated_tokens: List[int] = list(chain(*token_lists))

    # -------------------------------------------------------------------------
    # Step 4: Compute the total length and adjust to an exact multiple of block_size.
    total_length: int = len(concatenated_tokens)
    # Calculate the number of complete blocks that can be formed.
    num_blocks: int = total_length // block_size
    # Adjust total_length to discard any extra tokens that do not form a complete block.
    total_length = num_blocks * block_size

    # -------------------------------------------------------------------------
    # Step 5: Split the concatenated token list into blocks of fixed size.
    # The list comprehension iterates over the concatenated token list in steps of block_size,
    # slicing out a block each time.
    all_token_blocks: List[List[int]] = [
        concatenated_tokens[i: i + block_size] for i in range(0, total_length, block_size)
    ]

    # -------------------------------------------------------------------------
    # Step 6: Create labels for language modeling tasks.
    # It is common for language models to use the input_ids as labels.
    result: Dict[str, List[List[int]]] = {
        "input_ids": all_token_blocks,
        "labels": all_token_blocks.copy(),
    }

    return result


def main_train(config: dict = {}) -> str: # return final model path
    from datetime import datetime
    from transformers import TrainingArguments, Trainer
    from pathlib import Path
    import datetime
    import os
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )
    from datasets import load_dataset, Dataset
    from tfa import TfaCallback

    # Basic seeding
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # choose GPU
    seed = config.get('seed', 42)
    seed_everything(seed)

    # Log In
    from huggingface_hub import login, whoami
    key_file_path = "~/keys/master_hf_token.txt"
    key_file_path = os.path.abspath(os.path.expanduser(key_file_path))
    with open(key_file_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    login(token=token)
    os.environ['HUGGINGFACE_TOKEN'] = token
    user_info = whoami()
    print(f"Currently logged in as: {user_info['name']}\n")

    # Load model
    model_name = "google/gemma-2-2b"
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    today = config.get('today', datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss'))
    final_model_name: str = config.get('final_model_name', f'{model_name}-final-ft-model-{today}')

    # Prepare dataset
    # def my_prompt_format(nl_stmt: str) -> str:
    #     return (
    #         "Translate the natural language version of the mathematical statement "
    #         f"to a formal Lean version:\n{nl_stmt}\n"
    #     )
    def my_prompt_format(nl_stmt: str) -> str:
        # format iddah used for less: https://huggingface.co/datasets/AI4M/less-proofnet-lean4-top1M/viewer/default/train?row=0 
        return f'informal statement {nl_stmt}'
    # ds_train = load_dataset("UDACA/proofnet-v3-lean4", split="validation").with_format('torch') # Load the test split and convert data to PyTorch tensors for seamless integration with the training pipeline.
    ds_train = load_dataset("UDACA/proofnet-v3-lean4", split="test").with_format('torch') # Load the test split and convert data to PyTorch tensors for seamless integration with the training pipeline.
    ds_eval = load_dataset("UDACA/proofnet-v3-lean4", split="test").with_format('torch') # Load the test split and convert data to PyTorch tensors for seamless integration with the training pipeline.
    ds_tf_eval = load_dataset("UDACA/proofnet-v3-lean4", split="test").with_format('torch') # Load the test split and convert data to PyTorch tensors for seamless integration with the training pipeline.

    # Get string we want by filling the "text" field with the prompt we want
    # ds_train = ds_train.map(lambda eg: {"text": my_prompt_format(eg["nl_statement"]) + eg["formal_statement"]}, num_proc=24)
    # ds_train = ds_train.map(lambda egs: {"text": [my_prompt_format(eg["nl_statement"]) + eg["formal_statement"] for eg in egs]}, batched=True, remove_columns=remove_columns, num_proc=24)
    ds_train = ds_train.map(
            lambda batch: {
                "text": [
                    my_prompt_format(example["nl_statement"]) + example["formal_statement"]
                    for example in batch
                ]
            },
            batched=True,
            remove_columns=ds_train.column_names,  # remove all original columns
            num_proc=24,
    )
    # Map the function over your dataset. Here, 'text' is removed afterwards since it's no longer needed.
    ds_train = ds_train.map(
        lambda examples: tokenize_and_group_texts_via_blocks(examples, tokenizer=tokenizer, block_size=1024),
        batched=True,
        remove_columns=["text"],
        num_proc=24,
    )
    ds_eval = ds_eval.map(
        lambda examples: tokenize_and_group_texts_via_blocks(examples, tokenizer=tokenizer, block_size=1024),
        batched=True,
        remove_columns=["text"],
        num_proc=24,
    )
    # Get string data set needed for teacher forcing (we don't tokenize here due to crucial alignment correctness needed during tokenization implemented in the tf code)
    ds_tf_eval = ds_tf_eval.map(
        lambda egs: {
            'prompt': [my_prompt_format(eg['nl_statement']) for eg in egs], 
            'gold_response': [eg['formal_statement'] for eg in egs ]
        },
        num_proc=24
    )

    # 4) Minimal training args: run for 1 step, do evaluation at the same step.
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
        # save_strategy="no",
        # remove_unused_columns=False, # Not needed anymore because we have a seperate ds_tf_eval object
        save_steps=config.get('save_steps', 100), 
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
        # - push hub param for end of training
        push_to_hub=True,
        hub_model_id=final_model_name
    )

    # 5) Attach TfaCallback
    tfa_callback = TfaCallback(
        tfa_dataset=ds_tf_eval,
        repo=model_name,
        n_begin=186, # yes we want to do tf eval on full eval set
        n_during=4, # only partial tf eval during training, since it's per prompt, it would otherwise slow us down too much
        n_end=186 # yes we want to do tf eval on full eval set
    )

    # 6) Build trainer
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=ds_train,
       eval_dataset=ds_eval,  
       callbacks=[tfa_callback]
    )

    # 7) Run training
    trainer.train()

    # After training, explicitly save the final model and tokenizer to a separate folder.
    final_model_dir = output_dir / final_model_name
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))  # This saves both model and tokenizer if trainer.tokenizer is set.
    tokenizer.save_pretrained(str(final_model_dir))  # Extra precaution, if needed.

    # Push the final model to the Hub:
    trainer.push_to_hub(commit_message="Final model checkpoint", blocking=config.get('blocking', True))

    # return the final model path as str for later evals to use it if they want
    return str(final_model_dir)


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
