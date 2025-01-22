import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from typing import Optional

def seed_everything(seed: int = 42):
    import random
    import numpy as np
    from transformers import set_seed as hf_set_seed

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


def teacher_forced_tfa(
    prompt: str,
    solution: str,
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: str = "cuda"
) -> float:
    """
    Teacher-forced accuracy (token-level) on `solution` given a concatenated text = prompt + solution.

    Steps:
      1) Combined text = prompt + (some delimiter) + solution
      2) Tokenize combined text
      3) Forward pass -> logits
      4) Identify the token range for the solution
      5) Compare the predicted tokens in that range with the reference solution tokens
      6) Return fraction matched
    """
    # 1) Combine text
    combined_text = prompt + "\n\n" + solution

    # 2) Tokenize entire reference
    enc = tokenizer(combined_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # (1, total_seq_len)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # (1, total_seq_len, vocab_size)
    preds = torch.argmax(logits, dim=-1)  # (1, total_seq_len)

    # Tokenize the solution alone to find how many tokens are in it
    sol_enc = tokenizer(solution, add_special_tokens=False)
    solution_ids = sol_enc["input_ids"]
    len_solution = len(solution_ids)

    # Likewise, how many tokens in the prompt alone?
    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    len_prompt = len(prompt_enc["input_ids"])

    total_seq_len = input_ids.size(1)

    # If the combined text is too short or the solution doesn't fit, skip
    if len_prompt + len_solution >= total_seq_len:
        return 0.0

    # Teacher-forcing alignment:
    #   model's logit at position t => tries to predict token (t+1).
    # So the tokens for the solution start at index `len_prompt` in the combined text,
    # and we want to compare them against the model's predictions from
    #   [len_prompt : len_prompt+len_solution]
    #   to the reference tokens from
    #   [len_prompt+1 : len_prompt+1+len_solution].
    #
    # pred_slice: predicted tokens for the solution range
    pred_slice  = preds[:, len_prompt : len_prompt + len_solution]
    # label_slice: the actual solution tokens in combined_text
    label_slice = input_ids[:, (len_prompt + 1) : (len_prompt + 1) + len_solution]

    if pred_slice.size(1) == 0 or label_slice.size(1) == 0:
        return 0.0

    correctness = (pred_slice == label_slice).float()
    acc = correctness.mean().item()
    return acc


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # (Required) choose GPU
    seed_everything()

    # 1) Load the ProofNet validation set
    ds = load_dataset("hoskinson-center/proofnet", split="validation")

    # We'll just do the first N examples for demonstration
    # (Set N to len(ds) for the full dataset)
    N = 1  # you can increment this to see more
    sub_ds = ds.select(range(min(N, len(ds))))

    # 2) Our model list
    model_token_configs = [
        {
            "name": "GPT-2 (small)",
            "repo": "gpt2",  # small enough to test quickly
        },
        {
            "name": "Meta-Llama-3-8B-Instruct",
            "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        },
        # Add more as desired...
    ]

    # 3) Evaluate each model
    for config in model_token_configs:
        model_name = config["name"]
        repo = config["repo"]

        print(f"\nEvaluating {model_name} from {repo} on {N} example(s) of ProofNet validation.")
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

        # Some models (e.g. GPT-2) have no pad token => set it
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # We'll compute the average TFA across the sub-dataset
        sum_acc = 0.0
        for i, example in enumerate(sub_ds):
            nl_statement = example["nl_statement"]
            # prompt = f"Natural language version: {nl_statement}.\nTranslate the natural language version to a Lean version:\n"
            prompt = f"Natural language version. Translate the natural language version to a Lean version please:\n{nl_statement}\n"
            solution = example["formal_statement"]

            acc_i = teacher_forced_tfa(
                prompt=prompt,
                solution=solution,
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            sum_acc += acc_i
            print(f" Example {i}: TFA = {acc_i:.4f}")

        avg_tfa = sum_acc / N
        print(f" => Average TFA for {model_name} on these {N} example(s) = {avg_tfa:.4f}")


if __name__ == "__main__":
    main()
