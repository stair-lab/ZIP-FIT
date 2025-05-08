import os
import gc
from datetime import datetime
from pathlib import Path

from zip_fit.utils import seed_everything, save_final_model

import wandb

from zip_fit.nn_train.nn_model.model_and_tok import load_model_and_tok
from zip_fit.nn_train.trainer.train_data_src.prepare_train_data import prepare_datasets
from zip_fit.nn_train.trainer.prepare_trainer import create_trainer

def train(config: dict = {}) -> str:
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
