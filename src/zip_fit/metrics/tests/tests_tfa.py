import os
from typing import Dict, Any

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from zip_fit.metrics.tfa import tfa_teacher_forced_accuracy
from zip_fit.utils import seed_everything

def test_tfa(config: Dict[str, Any] = {}):
    # - Seed everything
    seed_everything(config.get('seed', 42))

    # - Load model and tokenizer
    model_name: str = config.get('model_name', 'meta-llama/Meta-Llama-3-8B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # - 
    prompt: str = "Question: 1 + 1 = ?\n\n"
    gold_response: str = "Solution: The answer is 2.\n\n"
    tfa: float = tfa_teacher_forced_accuracy(prompt, gold_response, model, model_name)
    print(f'{tfa=}')

if __name__ == "__main__":
    test_tfa()