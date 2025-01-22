import torch
import torch.nn as nn
from torch import tensor
from typing import List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
import os

def seed_everything(seed: int = 0, hf_timeout: float = 5) -> None:
    """
    Seed all necessary libraries to ensure reproducible results.
    Warning: Full reproducibility can still be tricky (e.g. different CUDA versions, OS setups, etc.).
    """
    import random
    import numpy as np
    from transformers import set_seed as hf_set_seed

    print(f"{seed=}")
    random.seed(seed)
    np.random.seed(seed)
    # Seed PyTorch (both CPU and CUDA if available)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Attempt to seed HF only on GPU (some versions can freeze if done on CPU).
    if torch.cuda.is_available():
        hf_set_seed(seed)
    else:
        print("Warning: HF is currently only deterministic/seeded when using GPU")

    # Try to seed vLLM
    try:
        from vllm import set_seed as vllm_set_seed
        vllm_set_seed(seed)
    except ImportError:
        print("vLLM not installed or vllm.set_seed has a bug, skipping vLLM seed setting.")


def compute_tfa(
    model: nn.Module,
    # tokenizer: PreTrainedTokenizer,
    repo: str,
    input_texts: List[str],
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None
) -> float:
    """
    Computes Teacher-Forced Accuracy (TFA) by feeding the model unshifted inputs,
    then comparing the model's position t with the label at position t+1.
    Specifically:
      - predictions_for_eval = predicted_token_ids[:, :-1]
      - labels_for_eval      = input_ids[:, 1:]

    We then ignore everything after the first real EOS in labels_for_eval.

    Parameters:
        model (nn.Module or PreTrainedModel): 
            A language model (Hugging Face CausalLM / decoder).
        tokenizer (PreTrainedTokenizer): 
            The tokenizer corresponding to the model.
        input_texts (List[str]): 
            List of input texts to compute TFA.
        bos_token_id (Optional[int]): 
            The BOS token ID. If None, we try tokenizer.bos_token_id, else fallback to eos_token_id.
        eos_token_id (Optional[int]): 
            The EOS token ID. If None, we try tokenizer.eos_token_id.
        pad_token_id (Optional[int]): 
            The PAD token ID. If None, we try tokenizer.pad_token_id, else fallback to eos_token_id.

    Returns:
        float: The TFA score (scalar).
               Ratio of (correctly predicted tokens up to and including the first EOS)
               over (total tokens up to first EOS), averaged across the batch.
    """
    # -------------------------------------------------
    # 1. Resolve BOS, EOS, PAD from arguments or fallback
    # -------------------------------------------------
    print(f'{repo=}')
    tokenizer = AutoTokenizer.from_pretrained(repo, padding_side="right", trust_remote_code=True) # note: add_eos_token not always implemented by tokenizer especially if trust_remote_code is True.
    if eos_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("EOS token is required but neither eos_token_id nor tokenizer.eos_token_id is set.")
        eos_token_id = tokenizer.eos_token_id

    if bos_token_id is None:
        if tokenizer.bos_token_id is not None:
            bos_token_id = tokenizer.bos_token_id
        else:
            bos_token_id = eos_token_id
            assert False

    if pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = eos_token_id

    # -------------------------------------------------
    # 2. Assign the pad token to the tokenizer
    # -------------------------------------------------
    tokenizer.pad_token_id = pad_token_id

    # -------------------------------------------------
    # 3. Tokenize the input texts (unshifted)
    # -------------------------------------------------
    input_texts = [input_text + tokenizer.eos_token for input_text in input_texts]
    inputs = tokenizer(input_texts, return_tensors='pt', padding=True, max_length=13, truncation=True)
    input_ids = inputs['input_ids']  # shape: (batch_size, seq_len)
    print(f'{input_ids=}')
    print(f'{tokenizer.batch_decode(input_ids)=}')
    # 'attention_mask' is ignored here, though you might use it for ignoring pads
    attention_mask = inputs['attention_mask']
    print(f'{attention_mask=}')
    # we need eos but to not go through all seqs let's check just first two to have eos present
    assert all(eos_token_id in input_id for input_id in input_ids[2:, :]), f'Input ids lacks eos token incorrectly: {input_ids=}'

    # -------------------------------------------------
    # 4. Forward pass (feed input_ids unshifted)
    # -------------------------------------------------
    # input_ids = input_ids[:, 1:]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

    # -------------------------------------------------
    # 5. Compute predictions (argmax)
    # -------------------------------------------------
    predicted_token_ids = torch.argmax(logits, dim=-1)  # shape: (batch_size, seq_len)

    # -------------------------------------------------
    # 6. Align predictions & labels for teacher forcing
    # -------------------------------------------------
    # Typical GPT logic: the logit at position t tries to guess token (t+1)
    # So we skip the last prediction and skip the first label to align them
    preds_for_eval = predicted_token_ids[:, :-1]  # shape: (batch_size, seq_len - 1)
    labels_for_eval = input_ids[:, 1:]           # shape: (batch_size, seq_len - 1)
    print(f'{input_ids.shape=}')
    print(f'{preds_for_eval.shape=}')
    print(f'{labels_for_eval.shape=}')

    print(f'{input_ids=}')
    print(f'{preds_for_eval=}')
    print(f'{labels_for_eval=}')

    print(f'{tokenizer.batch_decode(input_ids)=}')
    print(f'{tokenizer.batch_decode(preds_for_eval)=}')
    print(f'{tokenizer.batch_decode(labels_for_eval)=}')

    # -------------------------------------------------
    # 7. Find the first real EOS position in labels_for_eval
    # -------------------------------------------------
    # e.g. if labels_for_eval[i] = [..., eos_token_id, ...], we want that index
    eos_positions = (labels_for_eval == eos_token_id).int().argmax(dim=1)
    # shape: (batch_size,)

    # -------------------------------------------------
    # 8. Build a mask to ignore tokens after that EOS
    # -------------------------------------------------
    seq_len_minus1 = labels_for_eval.size(1)  # (seq_len - 1)
    indices = torch.arange(seq_len_minus1).unsqueeze(0).to(labels_for_eval.device)  # shape: (1, seq_len - 1)
    mask = indices <= eos_positions.unsqueeze(1)  # shape: (batch_size, seq_len - 1)

    # -------------------------------------------------
    # 9. Filter predictions and labels
    # -------------------------------------------------
    filtered_predictions = preds_for_eval[mask]
    filtered_labels = labels_for_eval[mask]

    # -------------------------------------------------
    # 10. Accuracy
    # -------------------------------------------------
    correct_predictions = (filtered_predictions == filtered_labels).float()
    accuracy = correct_predictions.mean().item()

    return accuracy


def main() -> None:
    """
    Example usage of compute_tfa.
    1) We seed everything for reproducibility.
    2) For each model, we:
        a) Load the model and tokenizer.
        b) Dynamically get bos_token_id, eos_token_id, and pad_token_id from the tokenizer.
        c) Compute TFA on some sample input_texts.
        d) We assert that TFA is in [0,1].
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'  # (Optional) GPU selection

    # 1) Seed everything
    seed_everything()

    model_token_configs = [
        # {
        #     "name": "google/codegemma-2b",
        #     "repo": "google/codegemma-2b",
        # },
        # {
        #     "name": "google/gemma-2-2b",
        #     "repo": "google/gemma-2-2b",
        # },
        # {
        #     "name": "internlm2-math-plus-1_8b",
        #     "repo": "internlm/internlm2-math-plus-1_8b",
        # },
        # {
        #     "name": "Mistral-7B-v0.1",
        #     "repo": "mistralai/Mistral-7B-v0.1",
        # },
        # {
        #     "name": "Meta-Llama-3-8B",
        #     "repo": "meta-llama/Meta-Llama-3-8B",
        # },
        # -- For debugging
        {
            "name": "Meta-Llama-3-8B-Instruct",
            "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        },
        # {
        #     "name": "google/gemma-2-2b-it",
        #     "repo": "google/gemma-2-2b-it"
        # }
    ]

    # Standard test inputs for TFA
    input_texts = [
        "The happy dog."
        # "The quick brown fox jumps over the lazy dog.",
        # "Artificial Intelligence is transforming the world of science."
    ]

    for config in model_token_configs:
        name = config["name"]
        repo = config["repo"]

        print(f"Evaluating {name} from {repo}")
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        # tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

        # Compute TFA
        tfa_score_general = compute_tfa(
            model=model,
            repo=repo,
            input_texts=input_texts,
        )
        # Ensure it's a valid probability
        assert 0.0 <= tfa_score_general <= 1.0, (
            f"TFA out of [0,1] range: {tfa_score_general}"
        )
        print(f"[{name}] TFA (General Inputs): {tfa_score_general:.4f}\n")


if __name__ == "__main__":
    main()
