import os
import time
from typing import Dict, Any

from datasets import load_dataset
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from zip_fit.metrics.tfa import compute_tfa_for_subds, tfa_teacher_forced_accuracy
from zip_fit.utils import seed_everything

def test_tfa(config: Dict[str, Any] = {}):
    # - Seed everything
    seed_everything(config.get('seed', 42))

    # - Load model and tokenizer
    model_name: str = config.get('model_name', 'meta-llama/Meta-Llama-3-8B-Instruct')
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
    # Move model to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # - 
    prompt: str = "Question: 1 + 1 = ?\n\n"
    gold_response: str = "Solution: The answer is 2.\n\n"
    tfa: float = tfa_teacher_forced_accuracy(prompt, gold_response, model, model_name, device=device)
    print(f'{tfa=}')

def test_tfa_with_proof_net():
    """
    Test TFA with the ProofNet dataset. This is the same sanity check we did 
    for the StackOverflow answer: https://stackoverflow.com/a/79379540/1601580 
    It seems we decided this was right because the TFA was none-zero if I remember correctly.
    But I'm not sure. It sanitcy check that definitively shows tfa is correct
    would be nice.

    wanbd run: https://wandb.ai/brando/zip-fit-tfa-tests/runs/sxi8y9w3/overview 
    """
    global_start_time = time.time()  # Start overall timer
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # choose GPU
    seed_everything()

    # 1) Load the ProofNet validation set
    ds = load_dataset("hoskinson-center/proofnet", split="validation")

    # load_datasetstom prompt format function
    def my_prompt_format(prompt: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{prompt}\n"
        )
    ds = ds.map(lambda example: {'prompt': 
                                 my_prompt_format(example['nl_statement']), 
                                 'gold_response': example['formal_statement']}, 
                                 num_proc=48)

    # We'll just do the first N examples for demonstration
    N = 5
    sub_ds = ds.select(range(min(N, len(ds))))

    # 2) Our model list (including all desired models, even if some remain commented)
    model_token_configs = [
        # {
        #     "name": "internlm2-math-plus-1_8b",
        #     "repo": "internlm/internlm2-math-plus-1_8b",
        # },
        # {
        #     "name": "google/gemma-2-2b",
        #     "repo": "google/gemma-2-2b",
        # },
        # {
        #     "name": "Mistral-7B-v0.1",
        #     "repo": "mistralai/Mistral-7B-v0.1",
        # },
        # {
        #     "name": "google/codegemma-2b",
        #     "repo": "google/codegemma-2b",
        # },
        # {
        #     "name": "GPT-2 (small)",
        #     "repo": "gpt2",
        # },
        {
            "name": "Meta-Llama-3-8B-Instruct",
            "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        },
        {
            "name": "Meta-Llama-3-8B",
            "repo": "meta-llama/Meta-Llama-3-8B",
        },
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for config in model_token_configs:
        model_name = config["name"]
        repo = config["repo"]

        print(f"\nEvaluating {model_name} from {repo} on {N} example(s) of ProofNet validation.")

        # Start per-model timer
        model_start_time = time.time()

        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).to(device)
        avg_tfa = compute_tfa_for_subds(
            sub_ds=sub_ds,
            model=model,
            repo=repo,
            device=device
        )

        # End per-model timer
        model_end_time = time.time()
        model_seconds = model_end_time - model_start_time

        print(f" => Average TFA for {model_name} on these {N} example(s) = {avg_tfa:.4f}")
        print(f" => Time to compute TFA for {model_name}: {model_seconds:.2f} seconds.")

    # End overall timer
    global_end_time = time.time()
    total_seconds = global_end_time - global_start_time
    print(f"\nDone. Total run time for all models: {total_seconds:.2f} seconds.")

def main():
    """ Runa all tests. """
    test_tfa()
    test_tfa_with_proof_net()

def _main(**kwargs):
    from datetime import datetime
    from socket import gethostname
    import wandb
    today = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss') # eg '2024_m01_d22_t13h_00m_30s'
    tmux_sess_num = None
    kwargs = kwargs | {'today': today, 'tmux_sess_num': tmux_sess_num, 'hostname': gethostname()}
    run_name = f'{kwargs}' 
    project: str = kwargs.get('project', 'zip-fit-tfa-tests')
    run = wandb.init(mode=kwargs.get('mode', 'dryrun'), project=project, name=run_name, save_code=True, config=kwargs)
    wandb.save(__file__) # save current code now, don't wait to wandb.finish, also useful: wandb.save("*.py") # upload all .py files in current directory
    print(f'Kwargs to run:\n{kwargs}')
    main()
    run.alert(title="Run Completed", text=f"Run finished, run url: {run.get_url()}")
    print(f'{run.get_url()=}')
    wandb.finish()

if __name__ == "__main__":
    import fire
    import time
    start_time = time.time()
    fire.Fire(_main)
    print(f"\aTime taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")


