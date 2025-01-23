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


def teacher_forced_accuracy_tfa(
    prompt: str,
    response: str,
    model: PreTrainedModel,
    repo: str,
    device: str = "cuda"
) -> float:
    """
    Teacher-forced accuracy (token-level) on `response` given a concatenated text = prompt + response.

    Steps:
      1) Combined text = prompt + (some delimiter) + response
      2) Tokenize combined text
      3) Forward pass -> logits
      4) Identify the token range for the response
      5) Compare the predicted tokens in that range with the reference response tokens
      6) Return fraction matched
    """
    # 1) Combine text
    combined_text = prompt + "\n\n" + response

    # X) Setup Tokenizer
    # since we are doign per row (prompt + resposne) processing, we don't need to worry if eos is not added (eg sometimes happens with llama 3 8b or trust_remote_code=True)
    # so wether the final eos is present or not will be consistently computed. No padding will be added since we aren't combining examples so no problem.
    # also no problem with bos, if it's added or not it won't make a difference since it's prefexied correctly already for tfa with the prompt. 
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # 2) Tokenize entire reference
    enc = tokenizer(combined_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # shape: (1, total_seq_len)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # shape: (1, total_seq_len, vocab_size)
    preds = torch.argmax(logits, dim=-1)  # shape: (1, total_seq_len)

    # Tokenize the response alone to find how many tokens are in it
    sol_enc = tokenizer(response, add_special_tokens=False)
    response_ids = sol_enc["input_ids"]
    len_response = len(response_ids)

    # Likewise, how many tokens in the prompt alone?
    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    len_prompt = len(prompt_enc["input_ids"])

    total_seq_len = input_ids.size(1)

    # If the combined text is too short or the response doesn't fit, skip
    if len_prompt + len_response >= total_seq_len:
        return 0.0

    # Teacher-forcing alignment:
    #   model's logit at position t => tries to predict token (t+1).
    # So the tokens for the response start at index `len_prompt` in the combined text,
    # and we want to compare them against the model's predictions from
    #   [len_prompt : len_prompt+len_response]
    #   to the reference tokens from
    #   [len_prompt+1 : (len_prompt+1)+len_response].
    #
    # pred_slice: predicted tokens for the response range
    pred_slice  = preds[:, len_prompt : len_prompt + len_response]
    # label_slice: the actual response tokens in combined_text
    label_slice = input_ids[:, (len_prompt + 1) : (len_prompt + 1) + len_response]

    if pred_slice.size(1) == 0 or label_slice.size(1) == 0:
        return 0.0

    correctness = (pred_slice == label_slice).float()
    acc = correctness.mean().item()
    return acc


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # (Required) choose GPU
    seed_everything()

    # 1) Load the ProofNet validation set
    ds = load_dataset("hoskinson-center/proofnet", split="validation")

    # We'll just do the first N examples for demonstration
    # (Set N to len(ds) for the full dataset)
    N = 1  # you can increment this to see more
    sub_ds = ds.select(range(min(N, len(ds))))

    # 2) Our model list (including all desired models, even if some stay commented, always evaluate all models don't remove):
    model_token_configs = [
        # - Absolutely neccessary for figure 5 for AutoFormalization
        {
            "name": "internlm2-math-plus-1_8b",
            "repo": "internlm/internlm2-math-plus-1_8b",
        },
        {
            "name": "google/gemma-2-2b",
            "repo": "google/gemma-2-2b",
        },
        {
            "name": "Mistral-7B-v0.1",
            "repo": "mistralai/Mistral-7B-v0.1",
        },
        # - For debugging & sanity checking the tfa code
        {
            "name": "google/codegemma-2b",
            "repo": "google/codegemma-2b",
        },
        {
            "name": "Meta-Llama-3-8B",
            "repo": "meta-llama/Meta-Llama-3-8B",
        },
        {
            "name": "Meta-Llama-3-8B-Instruct",
            "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        },
        {
            "name": "google/gemma-2-2b-it",
            "repo": "google/gemma-2-2b-it",
        },
        {
            "name": "GPT-2 (small)",
            "repo": "gpt2",  # small enough to test quickly
        },
        # You can reorder or comment out as needed
    ]

    # 3) Evaluate each model
    for config in model_token_configs:
        model_name = config["name"]
        repo = config["repo"]

        print(f"\nEvaluating {model_name} from {repo} on {N} example(s) of ProofNet validation.")
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        # note repo is always given to tfa not the tokenizer, since we need to 
        # control the exact way the tokenizer is set up and used for the tfa to be computed correctly & consistently accross model tokenizer's. 

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # We'll compute the average TFA across the sub-dataset
        sum_acc = 0.0
        for i, example in enumerate(sub_ds):
            nl_statement = example["nl_statement"]
            prompt = example["nl_statement"]
            prompt = f"Translate to formal Lean:\n{prompt}\n"
            prompt = f"Translate the natural language version of the mathematical statement to a Lean version please:\n{nl_statement}\n"
            response = example["formal_statement"]

            acc_i = teacher_forced_accuracy_tfa(
                prompt=prompt,
                response=response,
                model=model,
                repo=repo,
                device=device
            )
            sum_acc += acc_i
            print(f" Example {i}: TFA = {acc_i:.4f}")

        avg_tfa = sum_acc / N
        print(f" => Average TFA for {model_name} on these {N} example(s) = {avg_tfa:.4f}")


if __name__ == "__main__":
    main()
