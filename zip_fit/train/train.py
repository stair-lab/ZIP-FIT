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
) -> Dict[str, "torch.Tensor"]:
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
         blocks. Both fields are converted into PyTorch tensors with dtype torch.long.

    **Note on the attention mask field:**  
    The returned dictionary does not include an `attention_mask` field. Including an attention mask with a value of `None` 
    may cause issues with the Hugging Face Trainer, which expects either a valid tensor or for the key to be omitted entirely.

    Args:
        examples (Dict[str, List[str]]):
            A dictionary representing a batch of examples from the dataset. It must contain a key 'text'
            whose value is a list of strings (one per example).
        tokenizer (AutoTokenizer):
            A tokenizer instance (e.g., from Hugging Face's transformers) that can process a single text string.
            The tokenizer should return a dictionary containing at least the key 'input_ids' when called with a text,
            and it must have attributes `bos_token_id` (which may be None) and `eos_token_id`.
        block_size (int):
            The desired number of tokens per block (for example, 1024 or 4096).

    Returns:
        Dict[str, torch.Tensor]:
            A dictionary where:
              - 'input_ids' maps to a PyTorch tensor of token blocks, each block being a list of token IDs of length `block_size`
                and of dtype torch.long.
              - 'labels' maps to a copy of the 'input_ids' tensor.
            
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
    # pad_token: Optional[int] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

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
    # Convert the list of token blocks into PyTorch tensors with dtype=torch.long.
    import torch
    result: Dict[str, torch.Tensor] = {
        "input_ids": torch.tensor(all_token_blocks, dtype=torch.long),
        "labels": torch.tensor(all_token_blocks.copy(), dtype=torch.long),
    }

    return result


def main_train(config: dict = {}) -> str:
    """
    Trains the model using the provided configuration, saves the final checkpoint (model + tokenizer)
    in a dedicated subdirectory, and pushes the final model to the Hugging Face Hub.
    
    Args:
        config (dict): Configuration dictionary for training parameters.
        
    Returns:
        str: The final model directory path.
    """
    from datetime import datetime
    from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM
    from pathlib import Path
    import os
    import torch
    from datasets import load_dataset, Dataset
    from tfa import TfaCallback

    # ------------------------------
    # Set device and seed.
    # ------------------------------
    # export CUDA_VISIBLE_DEVICES=5; python ~/ZIP-FIT/zip_fit/train/train.py
    # export CUDA_VISIBLE_DEVICES=7; python ~/ZIP-FIT/zip_fit/train/train.py
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.get('cuda_visible_devices', '1')  # choose GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get('cuda_visible_devices', '3')  # choose GPU
    seed: int = config.get('seed', 42)
    seed_everything(seed)

    # ------------------------------
    # Log in to Hugging Face Hub.
    # ------------------------------
    from huggingface_hub import login, whoami
    key_file_path: str = os.path.abspath(os.path.expanduser(config.get('key_file_path', "~/keys/master_hf_token.txt")))
    with open(key_file_path, "r", encoding="utf-8") as f:
        token: str = f.read().strip()
    login(token=token)
    os.environ['HUGGINGFACE_TOKEN'] = token
    user_info = whoami()
    print(f"Currently logged in as: {user_info['name']}\n")

    # ------------------------------
    # Load model and tokenizer.
    # ------------------------------
    # model_name: str = config.get('model_name', 'gpt2')
    # model_name: str = config.get('model_name', 'Qwen/Qwen2.5-0.5B')
    # model_name: str = config.get('model_name', 'google/gemma-2-2b')
    # model_name: str = config.get('model_name', 'google/internlm2-math-plus-1_8b')
    model_name: str = config.get('model_name', 'Meta-Llama-3-8B')
    # model_name: str = config.get('model_name', 'google/codegemma-2b')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name).to(device) if 'gemma-2' not in model_name else AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager').to(device)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    today: str = config.get('today', datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss'))
    # Use a custom final model name if provided; otherwise use a default.
    final_model_name: str = config.get('final_model_name', f'UDACA/{model_name.replace("/", "-")}-pn-v3-lean4-train-on-validation-{today}')
    # final_model_name: str = config.get('final_model_name', f'UDACA/{model_name.replace("/", "-")}-pn-v3-lean4-train-on-test-{today}')

    final_model_name: str = config.get('final_model_name', f'AI4M/{model_name.replace("/", "-")}-pn-v3-lean4-train-on-validation-{today}')
    
    block_size: int = config.get('block_size', 1024)
    print(f'{block_size=}')

    # ------------------------------
    # Prepare datasets.
    # Three dataset views:
    #   - ds_train and ds_eval: tokenized for CE loss.
    #   - ds_tf_eval: raw strings for teacher-forced evaluation.
    # ------------------------------
    def my_prompt_format(nl_stmt: str) -> str:
        # Format used for less (see provided URL)
        return f'informal statement {nl_stmt}'

    # Load datasets with torch format.
    # Sanity Checks
    # ds_train: Dataset = load_dataset("UDACA/proofnet-v3-lean4", split="validation").with_format('torch')
    # ds_train: Dataset = load_dataset("UDACA/proofnet-v3-lean4", split="test").with_format('torch')

    # Lean4AI
    """
export CUDA_VISIBLE_DEVICES=3; python ~/ZIP-FIT/zip_fit/train/train.py --mode dryrun --project lean4ai-llama3-8b-runs --num_train_epochs 3 --model_name meta-llama/Meta-Llama-3-8B 
    """
    ds_train: Dataset = load_dataset("AI4M/mma-dataset", split="train").with_format('torch')
    # ds_train: Dataset = load_dataset("AI4M/stateInfoInformalizationBig", split="train").with_format('torch')
    # ds_train: Dataset = load_dataset("AI4M/regexInformalizationData", split="train").with_format('torch')
    # ds_train: Dataset = load_dataset("AI4M/gpt4-more", split="train").with_format('torch')

    # Evals
    ds_eval: Dataset  = load_dataset("UDACA/proofnet-v3-lean4", split="test").with_format('torch')
    ds_tf_eval: Dataset = load_dataset("UDACA/proofnet-v3-lean4", split="test").with_format('torch')

    # Create a new "text" field for tokenized datasets by concatenating formatted prompt and formal statement.
    ds_train = ds_train.map(
        lambda batch: {
            # Zip together the columns so we iterate over examples.
            "text": [my_prompt_format(nl) + formal 
                     for nl, formal in zip(batch["nl_statement"], batch["formal_statement"])]
        },
        batched=True,
        remove_columns=ds_train.column_names,  # Remove all original columns.
        num_proc=24,
    )
    print(f'{len(ds_train)=}')
    # Tokenize and group text for ds_train.
    ds_train = ds_train.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=ds_train.column_names, 
        # remove_columns=['text'], 
        num_proc=24,
    )
    print(f'{len(ds_train)=}')
    
    # Do the same for ds_eval.
    ds_eval = ds_eval.map(
        lambda batch: {
            # Zip together the columns so we iterate over examples.
            "text": [my_prompt_format(nl) + formal 
                     for nl, formal in zip(batch["nl_statement"], batch["formal_statement"])]
        },
        batched=True,
        remove_columns=ds_eval.column_names,  # Remove all original columns.
        num_proc=24,
    )
    ds_eval = ds_eval.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=ds_eval.column_names,
        num_proc=24,
    )
    # For teacher-forced evaluation, retain raw strings by creating 'prompt' and 'gold_response' fields.
    ds_tf_eval = ds_tf_eval.map(
        lambda batch: {
            'prompt': [my_prompt_format(nl) for nl in batch['nl_statement']],
            'gold_response': [formal for formal in batch['formal_statement']]
        },
        batched=True,
        num_proc=24
    )

    # ------------------------------
    # Define training arguments.
    # ------------------------------
    # Create the main output directory.
    output_dir: Path = Path(f'~/data/zipfit_less_runs/tfa_output_{today}').expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        # max_steps=2, # for debugging
        output_dir=str(output_dir),  # Main output directory.
        do_train=True,
        num_train_epochs=config.get('num_train_epochs', 3),  # Total training epochs.
        do_eval=True,
        eval_on_start=config.get('eval_on_start', True),     # Evaluate before training starts.
        evaluation_strategy=config.get('evaluation_strategy', "steps"),  # Evaluate every few steps.
        eval_steps=config.get('eval_steps', 1),             # Evaluate after every 25 steps.
        logging_steps=config.get('logging_steps', 1),       # Log metrics every 25 steps.
        per_device_train_batch_size=config.get('per_device_train_batch_size', 2),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        save_steps=config.get('save_steps', 25),            # Save a checkpoint every 100 steps.
        save_total_limit=config.get('save_total_limit', 1),    # Keep only the most recent checkpoint.
        save_strategy=config.get('save_strategy', 'steps'),
        bf16=torch.cuda.is_bf16_supported(),               # Use bf16 if supported.
        fp16=not torch.cuda.is_bf16_supported(),            # Otherwise, use fp16.
        # optim=config.get('optim', 'adamw_torch'),
        optim=config.get('optim', 'paged_adamw_32bit'),
        learning_rate=config.get('learning_rate', 1e-6),
        weight_decay=config.get('weight_decay', 1e-4),
        gradient_checkpointing=config.get('gradient_checkpointing', True), # careful, this can give issues depedning on hardware
        lr_scheduler_type=config.get('lr_scheduler_type', 'constant_with_warmup'),
        warmup_ratio=config.get('warmup_ratio', 0.05),
        seed=seed,
        data_seed=config.get('data_seed', seed),
        # torch_compile=True,
        # Hub push parameters.
        # push_to_hub=True,
        # hub_model_id=final_model_name,
    )

    # ------------------------------
    # Attach teacher-forced evaluation callback.
    # ------------------------------
    tfa_callback = TfaCallback(
        tfa_dataset=ds_tf_eval,
        repo=model_name,
        n_begin=config.get('n_begin', 184),  # Use full eval set at beginning.
        n_during=config.get('n_during', 4),    # Partial eval during training to save time.
        n_end=config.get('n_end', 184)         # Use full eval set at end.
    )

    # ------------------------------
    # Build the Trainer.
    # ------------------------------
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=ds_train,
       eval_dataset=ds_eval,
       callbacks=[tfa_callback],
       tokenizer=tokenizer,  # Ensure tokenizer is saved and pushed.
    )

    # ------------------------------
    # Run training.
    # ------------------------------
    trainer.train()

    # ------------------------------
    # Save final model and tokenizer in a dedicated subdirectory.
    # ------------------------------
    final_model_dir: Path = output_dir / final_model_name
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))          # Saves both model and tokenizer.
    tokenizer.save_pretrained(str(final_model_dir))     # Extra precaution.

    # ------------------------------
    # Push the final model (and tokenizer) to the Hub.
    # The push will be blocking by default unless 'blocking' is overridden in config.
    # ------------------------------
    trainer.push_to_hub(commit_message="Final model checkpoint", blocking=config.get('blocking', True))

    # Construct the final model URL and print it.
    final_model_url: str = f"https://huggingface.co/{final_model_name}"
    print(f"\nFinal model URL: {final_model_url}\n\n")

    # Clean
    del trainer

    import gc
    del model
    gc.collect()

    # Return the final model directory path as a string.
    results = {'final_model_url': final_model_url, 'final_model_dir': final_model_dir}
    return results

def main_full_run(config: dict = {}):
    from lean_pass_k_unbiased import run_pass_k_eval
    import os
    import wandb

    seed: int = config.get('seed', 42)
    seed_everything(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get('cuda_visible_devices', '1')  # choose GPU

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

    # Run params
    new_seed = config.get('seed', 42)
    k, N = 5, 6
    model_name: str = config.get('model_name', 'gpt2')
    # model_name: str = config.get('model_name', 'Qwen/Qwen2.5-0.5B')
    # model_name: str = config.get('model_name', 'google/gemma-2-2b')
    # model_name: str = config.get('model_name', 'google/internlm2-math-plus-1_8b')
    # model_name: str = config.get('model_name', 'Meta-Llama-3-8B')
    # model_name: str = config.get('model_name', 'google/codegemma-2b')

    # ---------------------------------------------------------------------
    # Initialize the Lean 4 server via PyPantograph.
    from pantograph import Server
    server = Server(imports=["Mathlib", "Init"], project_path=os.path.expanduser("~/mathlib4"))

    # ---------------------------------------------------------------------
    # Load prompts and gold headers from a dataset.
    # We assume that our prompt formatting function is defined below.
    def my_prompt_format(nl_stmt: str) -> str:
        return (
            "Your task is translate the natural language version of the mathematical statement "
            "to a formal Lean statement version, using the following format:\n"
            "natural language statement:\nLet $z=\\frac{1+i}{\\sqrt{2}}.$What is $\\left(z^{1^2}+z^{2^2}+z^{3^2}+\\dots+z^{{12}^2}\\right) \\cdot "
            "\\left(\\frac{1}{z^{1^2}}+\\frac{1}{z^{2^2}}+\\frac{1}{z^{3^2}}+\\dots+\\frac{1}{z^{{12}^2}}\\right)?$ "
            "$\\textbf{(A)}\\ 18 \\qquad \\textbf{(B)}\\ 72-36\\sqrt2 \\qquad \\textbf{(C)}\\ 36 \\qquad \\textbf{(D)}\\ 72 \\qquad \\textbf{(E)}\\ 72+36\\sqrt2$ "
            "Show that it is \\textbf{(C)}\\ 36.\n"
            "formal Lean language statement:##\ntheorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) : "
            "(∑ k in Finset.Icc 1 12, (z^(k^2))) * (∑ k in Finset.Icc 1 12, (1 / z^(k^2))) = 36 := sorry\n##"
            "natural language statement:\nIntegers $x$ and $y$ with $x>y>0$ satisfy $x+y+xy=80$. What is $x$? "
            "$\\textbf{(A)}\\ 8 \\qquad\\textbf{(B)}\\ 10 \\qquad\\textbf{(C)}\\ 15 \\qquad\\textbf{(D)}\\ 18 \\qquad\\textbf{(E)}\\ 26$ Show that it is \\textbf{(E)}\\ 26.\n"
            "formal Lean language statement:##\ntheorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) "
            "(h₂ : x + y + (x * y) = 80) : x = 26 := sorry\n##"
            "natural language statement:\nWhat is the [[volume]] of a [[cube]] whose [[surface area]] is twice that of a cube with volume 1? "
            "$\\mathrm{(A)}\\ \\sqrt{2}\\qquad\\mathrm{(B)}\\ 2\\qquad\\mathrm{(C)}\\ 2\\sqrt{2}\\qquad\\mathrm{(D)}\\ 4\\qquad\\mathrm{(E)}\\ 8$ Show that it is \\mathrm{(C)}.\n"
            "formal Lean language statement:##\ntheorem amc12a_2008_p8 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) "
            "(h₁ : y^3 = 1) (h₂ : 6 * x^2 = 2 * (6 * y^2)) : x^3 = 2 * Real.sqrt 2 := sorry\n##"
            f"natural language statement:\n{nl_stmt}\n"
            "formal Lean language statement:"
        )
    from datasets import load_dataset
    ds_test = load_dataset('UDACA/proofnet-v3-lean4', split='test')
    ds_test = ds_test.select(list(range(2)))  # optionally select a subset
    prompts = [my_prompt_format(row['nl_statement']) for row in ds_test]
    gold_headers = [row['header_no_import'] for row in ds_test]

    # Record the start time for this repetition.
    start_time = time.time()
    # Call run_pass_k_eval which:
    #   - Generates 'N' completions per prompt,
    #   - Checks each for correctness (e.g., compilation),
    #   - Computes and returns the overall pass@k score.
    pass_k_before_train = run_pass_k_eval(
        prompts=prompts,
        model_name=model_name,
        server=server,
        headers=gold_headers,
        k=k,
        num_samples=N,
        eval_batch_size=32,
        seed=new_seed,
        debug=False
    )
    # Record the end time and compute the elapsed time for this repetition.
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'{pass_k_before_train=}')
    print(f'{elapsed_time=}')
    wandb.log('pass_k_before_train', pass_k_before_train)

    # Train
    config = config | {'model_name': model_name}
    results = main_train(config)
    model_name = results['final_model_dir'] 

    # Record the start time for this repetition.
    start_time = time.time()
    # Call run_pass_k_eval which:
    #   - Generates 'N' completions per prompt,
    #   - Checks each for correctness (e.g., compilation),
    #   - Computes and returns the overall pass@k score.
    pass_k_before_train = run_pass_k_eval(
        prompts=prompts,
        model_name=model_name,
        server=server,
        headers=gold_headers,
        k=k,
        num_samples=N,
        eval_batch_size=32,
        seed=new_seed,
        debug=False
    )
    # Record the end time and compute the elapsed time for this repetition.
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'{pass_k_before_train=}')
    print(f'{elapsed_time=}')
    wandb.log('pass_k_before_train', pass_k_before_train)


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
    project: str = kwargs.get('project', 'zip-fit-train')
    # run = wandb.init(mode=kwargs.get('mode', 'online'), project="zip-fit-train", name=run_name, save_code=True, config=kwargs)
    run = wandb.init(mode=kwargs.get('mode', 'dryrun'), project=project, name=run_name, save_code=True, config=kwargs)
    wandb.save(__file__) # save current code now, don't wait to wandb.finish, also useful: wandb.save("*.py") # upload all .py files in current directory
    print(f'Kwargs to run:\n{kwargs}')
    # main(kwargs)
    main_train(kwargs)
    # main_full_run(kwargs)
    run.alert(title="Run Completed", text=f"Run finished, run url: {run.get_url()}")
    print(f'{run.get_url()=}')
    wandb.finish()

if __name__ == "__main__":
    import fire
    import time
    start_time = time.time()
    fire.Fire(_main)
    print(f"\aTime taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
