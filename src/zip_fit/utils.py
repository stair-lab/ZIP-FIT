import os
from pathlib import Path
from typing import List
from transformers import Trainer

def get_current_tmux_session_number() -> str:
    import os
    # Executes "tmux display-message -p '#S'" to retrieve the current tmux session name/number,
    # reads the output from the command, strips any surrounding whitespace, and returns it.
    return os.popen("tmux display-message -p '#S'").read().strip()

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
    # for vllm seed the object model: https://stackoverflow.com/questions/79467847/how-can-i-ensure-deterministic-text-generation-with-vllm-and-does-it-support-a/79467848#79467848

def generate_snippets_hf_pipeline(prompt: str, num_samples: int = 5, max_length: int = 512, model_name: str = 'gpt2') -> List[str]:
    """
    Generates Lean 4 code snippets from a textual prompt using GPT-2 (toy example).
    """
    from transformers import pipeline
    generator = pipeline("text-generation", model=model_name)
    outputs = generator(
        prompt,
        num_return_sequences=num_samples,  # e.g. 5 completions
        max_length=max_length,            # e.g. up to 64 tokens total
        do_sample=True,                   # random sampling
        temperature=1.0,                  # sampling "temperature", 1.0 => fairly random
    )
    return [o["generated_text"] for o in outputs] 


def login_to_huggingface(config: dict = {}) -> None:
    """
    Logs in to Hugging Face using the token stored in a file.
    """
    from huggingface_hub import login, whoami
    import os
    key_file_path: str = os.path.abspath(os.path.expanduser(config.get('key_file_path', "~/keys/master_hf_token.txt")))
    print(f"Logging in to Hugging Face using token from {key_file_path}")
    with open(key_file_path, "r", encoding="utf-8") as f:
        token: str = f.read().strip()
    login(token=token)
    os.environ['HUGGINGFACE_TOKEN'] = token
    user_info = whoami()
    print(f"Currently logged in as: {user_info['name']}\n")


def save_final_model(trainer: Trainer, final_model_name: str, output_dir: Path) -> Path:
    """ Saves the final model and tokenizer in a dedicated subdirectory. """
    final_model_dir: Path = output_dir / final_model_name
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer = trainer.tokenizer
    tokenizer.save_pretrained(str(final_model_dir))  
    return final_model_dir