import os
import time

import wandb

from pantograph import Server

from zip_fit.metrics.lean4_comp_pass_at_k import lean4_comp_pass_at_k_unbiased
from zip_fit.nn_train.train import train
from zip_fit.eval.eval_data_src.eval_lean4_prepare_data import prepare_eval_data
from zip_fit.utils import login_to_huggingface, seed_everything

def main_train_eval(config: dict = {}):
    """ Trains a model and evaluates it on a dataset of ProofNet problems (Lean 4). """
    # - Seed everything
    seed: int = config.get('seed', 42)
    seed_everything(seed)
    # - Login to Hugging Face
    login_to_huggingface(config)

    # - Initialize the Lean 4 server via PyPantograph
    server = Server(imports=["Mathlib", "Init"], project_path=os.path.expanduser("~/mathlib4"))

    # - Load prompts & gold headers then evaluate pass@k before training
    prompts, gold_headers, ds_test = prepare_eval_data(config)
    start_time = time.time()
    pass_k_before_train: float = lean4_comp_pass_at_k_unbiased(
        prompts=prompts,
        model_name=config.get('model_name', 'meta-llama/Meta-Llama-3-8B-Instruct'),
        server=server,
        headers=gold_headers,
        k=config.get('k', 5),
        num_samples=config.get('N', 200),
        eval_batch_size=config.get('eval_batch_size', 32),
        seed=seed,
        debug=False
    )
    print(f'Result before training: {pass_k_before_train=}\n Time: {time.time() - start_time}')
    wandb.log('pass_k_before_train', pass_k_before_train)

    # - Train
    results: dict = train(config)
    model_name: str = results['final_model_dir'] 

    # - Evaluate pass@k after training
    start_time = time.time()
    pass_k_after_train: float = lean4_comp_pass_at_k_unbiased(
        prompts=prompts,
        model_name=model_name,
        server=server,
        headers=gold_headers,
        k=config.get('k', 5),
        num_samples=config.get('N', 200),
        eval_batch_size=config.get('eval_batch_size', 32),
        seed=seed,
        debug=False
    )
    print(f'Result after training: {pass_k_after_train=}\n Time: {time.time() - start_time}')
    wandb.log('pass_k_after_train', pass_k_after_train)

    # - Return results
    results = results | {'pass_k_before_train': pass_k_before_train, 'pass_k_after_train': pass_k_after_train}
    return results

def _main(**kwargs):
    from datetime import datetime
    from socket import gethostname
    import wandb
    today = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss') # eg '2024_m01_d22_t13h_00m_30s'
    tmux_sess_num = None
    kwargs = kwargs | {'today': today, 'tmux_sess_num': tmux_sess_num, 'hostname': gethostname()}
    run_name = f'{kwargs}' 
    project: str = kwargs.get('project', 'zip-fit-train')
    run = wandb.init(mode=kwargs.get('mode', 'dryrun'), project=project, name=run_name, save_code=True, config=kwargs)
    wandb.save(__file__) # save current code now, don't wait to wandb.finish, also useful: wandb.save("*.py") # upload all .py files in current directory
    print(f'Kwargs to run:\n{kwargs}')
    main_train_eval(kwargs)
    run.alert(title="Run Completed", text=f"Run finished, run url: {run.get_url()}")
    print(f'{run.get_url()=}')
    wandb.finish()

if __name__ == "__main__":
    import fire
    import time
    start_time = time.time()
    fire.Fire(_main)
    print(f"\aTime taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
