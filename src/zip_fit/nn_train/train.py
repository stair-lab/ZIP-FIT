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
from zip_fit.nn_train.train_data_src.prepare_train_data import prepare_datasets
from zip_fit.nn_train.trainer.prepare_trainer import create_trainer

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
    
    # - Prepare datasets using the dedicated module
    dataset_type = config.get("dataset_type", "zipfit/math-select-06062025")
    ds_train, ds_eval, ds_tf_eval = prepare_datasets(
        tokenizer=tokenizer,
        config=config,
        dataset_type=dataset_type
    )

    # - Create trainer using the dedicated module
    trainer, output_dir = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tf_eval_dataset=ds_tf_eval,
        model_name=model_name,
        config=config,
        seed=seed
    )

    # - Run training.
    trainer.train()

    # - Save final model and tokenizer in a dedicated subdirectory.
    final_model_dir: Path = output_dir / final_model_name
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))          # Saves both model and tokenizer.
    tokenizer.save_pretrained(str(final_model_dir))     # Extra precaution.

    # - Push the final model (and tokenizer) to the Hub.
    # The push will be blocking by default unless 'blocking' is overridden in config.
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
