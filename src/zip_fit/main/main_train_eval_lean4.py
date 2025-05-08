
import os
import time

from zip_fit.eval.eval_data_src.eval_lean4_prepare_data import prepare_eval_data
from zip_fit.utils import login_to_huggingface, seed_everything

from pantograph import Server

def main_train_eval(config: dict = {}):
    """
    Trains a model and evaluates it on a dataset of ProofNet problems (Lean 4).
    """
    # - Seed everything
    seed: int = config.get('seed', 42)
    seed_everything(seed)
    # - Login to Hugging Face
    login_to_huggingface(config)

    # Initialize the Lean 4 server via PyPantograph.
    server = Server(imports=["Mathlib", "Init"], project_path=os.path.expanduser("~/mathlib4"))

    # - Load prompts and gold headers from a dataset.
    prompts, gold_headers, ds_test = prepare_eval_data(config)
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
        seed=seed,
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
    results = train(config)
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
        seed=seed,
        debug=False
    )
    # Record the end time and compute the elapsed time for this repetition.
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'{pass_k_before_train=}')
    print(f'{elapsed_time=}')
    wandb.log('pass_k_before_train', pass_k_before_train)