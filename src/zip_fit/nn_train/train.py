import gc
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import Dataset
from zip_fit.eval.eval_data_src.prepare_eval_tf_datasets import get_eval_tf_datasets
from zip_fit.nn_train.trainer.train_data_src.prepare_train_datasets import get_train_datasets
from zip_fit.utils import seed_everything, save_final_model

from zip_fit.nn_train.nn_model.model_and_tok import load_model_and_tok
from zip_fit.nn_train.trainer.prepare_trainer import create_trainer

from transformers import AutoModelForCausalLM, AutoTokenizer

def train(
        # model: Optional[AutoModelForCausalLM] = None, 
        # tokenizer: Optional[AutoTokenizer] = None, 
        # ds_tf_eval: Optional[Dataset] = None, 
        config: dict = {}) -> str:
    """
    Trains the model, saves the final checkpoint (model + tokenizer) in a dedicated subdirectory, 
    and pushes the final model to the Hugging Face Hub.
    """
    # - Set device and seed
    seed: int = config.get('seed', 42)
    seed_everything(seed)

    # - Load model and tokenizer
    model_name: str = config.get('model_name', 'meta-llama/Meta-Llama-3-8B-Instruct')
    model, tokenizer = load_model_and_tok(config) 
    
    # - Prepare datasets using the appropriate specialized module
    ds_train = get_train_datasets(tokenizer, 
                                  dataset_name=config.get("training_dataset_name", "zipfit/math-select-06062025"), 
                                  split=config.get("training_split", "train"), config=config)
    ds_eval = get_train_datasets(tokenizer, 
                                 dataset_name=config.get("training_eval_dataset_name", "Putnam-AXIOM-for-zip-fit-splits"), 
                                 split=config.get("training_eval_split", "validation"), config=config)
    ds_tf_eval = get_eval_tf_datasets(tokenizer, split=config.get("training_tf_eval_split", "validation"), config=config)
    
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
    today: str = config.get('today', datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss'))
    final_model_name: str = config.get('final_model_name', f'{model_name}') + f'-{today}'
    final_model_dir = save_final_model(trainer, final_model_name, output_dir)

    # - Push the final model (and tokenizer) to the Hub (blocking by default)
    trainer.push_to_hub(commit_message="Final model checkpoint", blocking=config.get('blocking', True))

    # Construct the final model URL and print it.
    final_model_url: str = f"https://huggingface.co/{final_model_name}"
    print(f"\nFinal model URL: {final_model_url}\n\n")

    # Clean up (because we might use vllm after this function call is done)
    del trainer
    del model
    gc.collect()

    # Return the final model directory path as a string.
    results = {'final_model_url': final_model_url, 'final_model_dir': str(final_model_dir)}
    print(f'{results}')
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
    train(kwargs)
    run.alert(title="Run Completed", text=f"Run finished, run url: {run.get_url()}")
    print(f'{run.get_url()=}')
    wandb.finish()

if __name__ == "__main__":
    import fire
    import time
    start_time = time.time()
    fire.Fire(_main)
    print(f"\aTime taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
