# utils.py

import os
from typing import List

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

# TODO: unsure if needed...
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