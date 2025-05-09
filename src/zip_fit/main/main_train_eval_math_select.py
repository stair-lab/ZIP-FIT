import os
import time

import wandb

from zip_fit.nn_train.train import train
from zip_fit.utils import login_to_huggingface, seed_everything

def main_train_eval(config: dict = {}):
    """ Trains a model and evaluates it on a dataset of ProofNet problems (Lean 4). """
    # - Seed everything
    seed: int = config.get('seed', 42)
    seed_everything(seed)
    # - Login to Hugging Face
    login_to_huggingface(config)

    # Prepare Eval dataset
    ds_eval, ds_tf_eval = prepare_math_eval_datasets(tokenizer, config, dataset_name)

    # - Evaluate math_acc before training
    math_acc_before_train: float = math_acc_before_train(config)
    print(f'Result before training: {math_acc_before_train=}\n Time: {time.time() - start_time}')
    wandb.log('math_acc_before_train', math_acc_before_train)

    # - Evaluate log likelihood gold ref before training 
    compute_log_likelihood_for_subds()

    # - Train
    results: dict = train(config)
    model_name: str = results['final_model_dir'] 

    # - Evaluate pass@k after training
    start_time = time.time()
    math_acc_after_train: float = math_acc_after_train(config)
    print(f'Result after training: {math_acc_after_train=}\n Time: {time.time() - start_time}')
    wandb.log('math_acc_after_train', math_acc_after_train)

    # - Return results
    results = results | {'math_acc_before_train': math_acc_before_train, 'math_acc_after_train': math_acc_after_train}
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
