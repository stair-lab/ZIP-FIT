
import os
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import torch
from datasets import load_dataset, Dataset

from zip_fit.utils import seed_everything

from tfa import TfaCallback

from zip_fit.nn_train.nn_train_utils import tokenize_and_group_texts_via_blocks
from zip_fit.nn_train.nn_model.model_and_tok import load_model_and_tok

def main_train(config: dict = {}) -> str:
    """
    Trains the model using the provided configuration, saves the final checkpoint (model + tokenizer)
    in a dedicated subdirectory, and pushes the final model to the Hugging Face Hub.
    """
    # - Set device and seed
    seed: int = config.get('seed', 42)
    seed_everything(seed)

    # - Load model and tokenizer
    model_name: str = config.get('model_name', 'meta-llama/Meta-Llama-3-8B-Instruct')
    model, tokenizer = load_model_and_tok(config)

    today: str = config.get('today', datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss'))
    final_model_name: str = config.get('final_model_name', f'{model_name.replace("/", "-")}-{today}')
    

    # ---

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
    """
export CUDA_VISIBLE_DEVICES=4; python ~/ZIP-FIT/zip_fit/train/train.py --mode dryrun --project zip-fit-pn-sanity --num_train_epochs 3 --model_name meta-llama/Meta-Llama-3-8B 
    """
    # ds_train: Dataset = load_dataset("UDACA/proofnet-v3-lean4", split="test").with_format('torch')
    # ds_train: Dataset = load_dataset("UDACA/proofnet-v3-lean4", split="validation").with_format('torch')

    # ds_eval: Dataset  = load_dataset("UDACA/proofnet-v3-lean4", split="test").with_format('torch')
    # ds_tf_eval: Dataset = load_dataset("UDACA/proofnet-v3-lean4", split="test").with_format('torch')

    # Lean4AI
    """
export CUDA_VISIBLE_DEVICES=4; python ~/ZIP-FIT/zip_fit/train/train.py --mode dryrun --project lean4ai-llama3-8b-runs --num_train_epochs 3 --model_name meta-llama/Meta-Llama-3-8B 
    """
    # ds_train: Dataset = load_dataset("AI4M/mma-dataset", split="train").with_format('torch')
    # ds_train: Dataset = load_dataset("AI4M/stateInfoInformalizationBig", split="train").with_format('torch')
    # ds_train: Dataset = load_dataset("AI4M/regexInformalizationData", split="train").with_format('torch')
    # ds_train: Dataset = load_dataset("AI4M/gpt4-more", split="train").with_format('torch')
    # ds_train: Dataset = load_dataset("AI4M/gpt4-more", split="train").with_format('torch')

    """
conda activate zip_fit
conda activate zip_fit
export CUDA_VISIBLE_DEVICES=4

python ~/ZIP-FIT/zip_fit/train/train.py --mode online --project self-opt-train-uncompiled-py-2-gsm8k --num_train_epochs 1 --model_name gpt2
python ~/ZIP-FIT/zip_fit/train/train.py --mode dryrun --project self-opt-train-uncompiled-py-2-gsm8k --num_train_epochs 1 --model_name Qwen/Qwen2.5-0.5B

python ~/ZIP-FIT/zip_fit/train/train.py --mode dryrun --project self-opt-train-uncompiled-py-2-gsm8k --num_train_epochs 1 --model_name google/gemma-2-2b 

python ~/ZIP-FIT/zip_fit/train/train.py --mode dryrun --project self-opt-train-uncompiled-py-2-gsm8k --num_train_epochs 1 --model_name meta-llama/Llama-3.2-1B 
python ~/ZIP-FIT/zip_fit/train/train.py --mode dryrun --project self-opt-train-uncompiled-py-2-gsm8k --num_train_epochs 1 --model_name meta-llama/Llama-3.2-3B 

python ~/ZIP-FIT/zip_fit/train/train.py --mode dryrun --project self-opt-train-uncompiled-py-2-gsm8k --num_train_epochs 1 --model_name meta-llama/Meta-Llama-3-8B 
    """
    def my_prompt_format(question: str, answer: str, final_answer: str)-> str:
        return f'question: {question}\nanswer: {answer}\n### {final_answer}'
    # Path to your saved JSON file
    # ds_train = load_dataset("json", data_files=os.path.expanduser("~/data/synthetic_data/uncompiled_dspy/syndata_100_2025_m02_d05_t18h_29m_01s.json"))
    ds_train = load_dataset("json", split='train[:70]', data_files=os.path.expanduser("~/data/synthetic_data/uncompiled_dspy/syndata_18612_2025_m02_d05_t18h_38m_11s.json"))

    # Evals
    ds_eval: Dataset  = load_dataset("openai/gsm8k", 'main', split="test").with_format('torch')
    ds_tf_eval: Dataset = load_dataset("openai/gsm8k", 'main', split="test").with_format('torch')

    # Create a new "text" field for tokenized datasets by concatenating formatted prompt and formal statement.
    ds_train = ds_train.map(
        # lambda batch: {"text": [my_prompt_format(nl) + formal for nl, formal in zip(batch["y_grade_school_math_question"], batch["y_grade_school_math_solution"], batch["y_grade_school_math_final_answer"])]},
        lambda batch: {"text": [my_prompt_format(q, a, fa) for q, a, fa in zip(batch["y_grade_school_math_question"], batch["y_grade_school_math_solution"], batch["y_grade_school_math_final_answer"])]},
        batched=True,
        remove_columns=ds_train.column_names,  # Remove all original columns.
        num_proc=48,
    )
    print(f'{len(ds_train)=} {ds_train.column_names=}')
    # Tokenize and group text for ds_train.
    ds_train = ds_train.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=ds_train.column_names, 
        # remove_columns=['text'], 
        num_proc=48,
    )
    print(f'{len(ds_train)=} {ds_train.column_names=}')
    
    # Do the same for ds_eval.
    ds_eval = ds_eval.map(
        # lambda batch: {"text": [my_prompt_format(nl) + formal for nl, formal in zip(batch["nl_statement"], batch["formal_statement"])]},
        lambda batch: {"text": [f'question: {q}\nanswer: {a}' for q, a in zip(batch["question"], batch["answer"])]},
        batched=True,
        remove_columns=ds_eval.column_names,  # Remove all original columns.
        num_proc=48,
    )
    print(f'{len(ds_eval)=} {ds_eval.column_names=}')
    ds_eval = ds_eval.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
        remove_columns=ds_eval.column_names,
        num_proc=48,
    )
    print(f'{len(ds_eval)=} {ds_eval.column_names=}')
    # For teacher-forced evaluation, retain raw strings by creating 'prompt' and 'gold_response' fields.
    ds_tf_eval = ds_tf_eval.map(
        lambda batch: {
            # 'prompt': [my_prompt_format(nl) for nl in batch['nl_statement']],
            # 'gold_response': [formal for formal in batch['formal_statement']]
            'prompt': [f'question: {q}\nanswer: ' for q in batch['question']],
            'gold_response': [answer for answer in batch['answer']]
        },
        batched=True,
        num_proc=48
    )
    print(f'{len(ds_tf_eval)=} {ds_tf_eval.column_names=}')

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
        # num_train_epochs=config.get('num_train_epochs', 3),  # Lean4AI Total training epochs.
        num_train_epochs=config.get('num_train_epochs', 1),  # Total training epochs.
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
        # remove_unused_columns=False,
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
    print(f'{results}')
    return results

def main_full_run(config: dict = {}):
    from metrics.lean4_comp_pass_at_k import run_lean4_comp_pass_k_unbiased_eval
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
    pass_k_before_train = run_lean4_comp_pass_k_unbiased_eval(
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
    pass_k_before_train = run_lean4_comp_pass_k_unbiased_eval(
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
