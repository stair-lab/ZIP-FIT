from datetime import datetime
import torch
from trl import SFTConfig, SFTTrainer
import os
import fire
import wandb

import train
from train.tfa_callback import TfaCallback 

from tfa import seed_everything

def get_current_tmux_session_number() -> str:
    """ Returns the current tmux session number. """
    import subprocess
    try:
        # 'tmux display-message -p "#S"' gets the current session's name/number.
        output = subprocess.check_output(['tmux', 'display-message', '-p', '#S'], text=True)
        return output.strip()
    except Exception:
        return ""

def main(**config):
    print(f'Config for main\'s run:\n{config}')

    # -- Seed everything
    seed_everything(seed=config.get('seed', 0))
    
    # -- HF login
    from huggingface_hub import login
    token = open(os.path.expanduser("~/keys/master_hf_token.txt")).read().strip()
    login(token=token)

    # -- Get model
    repo = config.get('pretrained_model_name_or_path', 'internlm/internlm2-math-plus-1_8b')
    # repo = config.get('pretrained_model_name_or_path', 'google/gemma-2-2b')
    # repo = config.get('pretrained_model_name_or_path', 'mistralai/Mistral-7B-v0.1')
    model, tok = train.models.load_model_and_tok(repo, config) 

    # -- Load datasets
    ds_name_or_path = config.get('ds_name_or_path', 'hoskinson-center/proofnet')
    train_split, val_split = config.get('train_split', 'validation'), config.get('val_split', 'test')
    print(f'\n---> {ds_name_or_path=} {train_split=} {val_split=}\n')
    train_dataset = load_proofnet(ds_name_or_path, tok, config.get('max_seq_length', 512), config, model, end=config.get('end_train', 1), split=train_split)
    print_first_example_after_decode(train_dataset, tok)

    eval_dataset = load_math_style_dataset(ds_name_or_path, tok, config.get('max_seq_length', 512), config, end=36, split=val_split)
    print(f'{len(train_dataset)=}\n{len(eval_dataset)=}')
    wandb.config.update({'dataset': f'{ds_name_or_path} ({train_split=} {val_split=})'})

    # -- Prepare output directory
    today: str = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss')
    output_dir: str = os.path.expanduser(f"~/data/runs_logic_cont/run_{config.get('today', today)}")
    print(f'{output_dir=}')
    
    # -- Train model
    # - Prepare output directory
    today: str = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss')
    output_dir: str = os.path.expanduser(f"~/data/runs_logic_cont/run_{config.get('today', today)}")
    print(f'{output_dir=}')
    training_args = SFTConfig(
        output_dir=output_dir,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        # -- save, eval, logging opts
        # save_steps=config.get('save_steps', 2), 
        save_total_limit=3,
        save_strategy=config.get('save_strategy', 'steps'),
        eval_on_start=config.get('eval_on_start', True),
        evaluation_strategy=config.get('eval_strategy', 'steps'), 
        eval_steps=config.get('eval_steps', 1), 
        logging_first_step=config.get('logging_first_step', True), # Default to False, unsure 100% what this does but looks like a good idea
        logging_strategy=config.get('logging_strategy', 'steps'),
        logging_steps=config.get('logging_steps', 1),
        # -- 
        max_steps=config.get('max_steps', 12),
        num_train_epochs=config.get('num_train_epochs', 10),
        max_seq_length=config.get('max_seq_length', 512),
        per_device_train_batch_size=config.get('batch_size', 2),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        # -- optim
        optim=config.get('optim', 'adamw_torch'),
        # optim=config.get('optim', 'paged_adamw_32bit'),
        learning_rate=config.get('learning_rate', 1e-5),
        weight_decay=config.get('weight_decay', 1e-4),
        gradient_checkpointing=config.get('gradient_checkpointing', False), # careful might give issues, but not in skampere1
        # -- scheduler
        lr_scheduler_type=config.get('lr_scheduler_type', 'constant'), # this is the hf default btw
        warmup_ratio=config.get('warmup_ratio', 0.0), 
        # -- seed
        seed=config.get('seed', 0),
        data_seed=config.get('data_seed', config.get('seed', 0)),
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[TfaCallback(model, tok)]
    )
    print(f"\nStarting fine-tuning...")
    print(f'If traning from scratch, expected initial loss (roughly): {torch.log(torch.tensor(len(tok.vocab)))=}')
    # - Save the initial model and tokenizer as checkpoint-0
    initial_checkpoint_dir = os.path.join(output_dir, "checkpoint-0")
    os.makedirs(initial_checkpoint_dir, exist_ok=True)
    print(f"Saving initial checkpoint and tokenizer at {initial_checkpoint_dir}")
    model.save_pretrained(initial_checkpoint_dir)
    tok.save_pretrained(initial_checkpoint_dir)
    # - Train
    trainer.train()
    # - end run
    return os.path.expanduser(output_dir)

def _main(**kwargs):
    from datetime import datetime
    import os
    from socket import gethostname
    today = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss') # eg '2024_m01_d22_t13h_00m_30s'
    tmux_sess_num: str = get_current_tmux_session_number()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(kwargs.get('CUDA_VISIBLE_DEVICES', '7'))
    print(f'Current Tmux Session Number: {tmux_sess_num}')
    kwargs = kwargs | {'today': today, 'tmux_sess_num': tmux_sess_num, 'hostname': gethostname()}
    run_name = f'{kwargs}' 
    run = wandb.init(mode=kwargs.get('mode', 'dryrun'), project="putnam-axiom", name=run_name, save_code=True, config=kwargs)
    # run = wandb.init(mode=kwargs.get('mode', 'online'), project="putnam-axiom", name=run_name, save_code=True, config=kwargs)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str('2,3'); print("--> WARNING/REMINDER: cude device harcoded in script!\n"*10)
    print(f'Kwargs to run:\n{kwargs}')
    output_dir = main(**kwargs)
    run_eval_logic_contamination(output_dir)
    run.alert(title="Run Completed", text=f"Run finished, run url: {run.get_url()}")
    print(f'{run.get_url()=}')
    wandb.finish()

if __name__ == "__main__":
    import time
    start_time = time.time()
    fire.Fire(_main)
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
