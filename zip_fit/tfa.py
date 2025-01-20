import torch
import torch.nn as nn
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
        torch.cuda.manual_seed_all(seed)  # If you use multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Attempt to seed HF only on GPU (some versions can freeze if done on CPU).
    if torch.cuda.is_available():
        hf_set_seed(seed)  # occasionally can cause halting on some systems
    else:
        print("Warning: HF is currently only deterministic/seeded when using GPU")

    # Try to seed vllm
    try:
        from vllm import set_seed as vllm_set_seed
        vllm_set_seed(seed)
    except ImportError:
        print("vLLM not installed or vllm.set_seed has a bug, skipping vLLM seed setting.")


def compute_tfa(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    input_texts: List[str],
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None
) -> float:
    """
    Computes Teacher-Forced Accuracy (TFA), rewarding the model for correctly predicting
    the first EOS token while ignoring predictions for padding tokens.

    We handle the edge case where BOS == EOS by temporarily masking out the first token
    so it doesn't register as an 'early EOS' at index=0.

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
            The EOS token ID. If None, we try tokenizer.eos_token_id. This is needed for correct masking.
        pad_token_id (Optional[int]): 
            The PAD token ID. If None, we try tokenizer.pad_token_id, else fallback to eos_token_id.

    Returns:
        float: The TFA score (scalar). 
            Ratio of (correctly predicted tokens up to and including the first real EOS) 
            over (total tokens up to first EOS), averaged across the batch.
    """

    # -------------------------------------------------
    # 1. Resolve BOS, EOS, PAD from arguments or fallback
    # -------------------------------------------------
    if eos_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("EOS token is required but neither eos_token_id nor tokenizer.eos_token_id is set.")
        eos_token_id = tokenizer.eos_token_id

    if bos_token_id is None:
        if tokenizer.bos_token_id is not None:
            bos_token_id = tokenizer.bos_token_id
        else:
            bos_token_id = eos_token_id

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
    # 3. Tokenize the input texts
    # -------------------------------------------------
    inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
    # 'inputs' -> 'input_ids': shape: (batch_size, seq_len), 'attention_mask': (batch_size, seq_len)
    input_ids = inputs['input_ids']  # shape: (batch_size, seq_len)

    # -------------------------------------------------
    # 4. Create right-shifted input by adding BOS at the start if BOS token not present already
    # -------------------------------------------------
    if input_ids[0, 0] != bos_token_id: 
        right_shifted_input_ids = torch.cat([
            torch.full((input_ids.shape[0], 1), bos_token_id, dtype=torch.long),  # shape: (batch_size, 1)
            input_ids[:, :-1]                                                     # shape: (batch_size, seq_len - 1)
        ], dim=1)  # shape: (batch_size, seq_len)
    else:
        right_shifted_input_ids = input_ids  # I don't think clone is needed since we never modify right shifted toks anyway

    # -------------------------------------------------
    # 5. Forward pass with the right-shifted inputs
    # -------------------------------------------------
    with torch.no_grad():
        outputs = model(input_ids=right_shifted_input_ids)
        # outputs.logits: (batch_size, seq_len, vocab_size)

    logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

    # -------------------------------------------------
    # 6. Compute predictions
    # -------------------------------------------------
    predicted_token_ids = torch.argmax(logits, dim=-1)  # shape: (batch_size, seq_len)
    print(f'{predicted_token_ids=}')
    print(f'{input_ids=}')
    print(f'{right_shifted_input_ids=}')
    
    print()
    print(f'{predicted_token_ids.shape=}')
    print(f'{input_ids.shape=}')
    print(f'{right_shifted_input_ids.shape=}')

    # -------------------------------------------------
    # 7. Find the first *real* EOS position in each sequence
    # -------------------------------------------------
    # If BOS == EOS, the model might treat token at index=0 as "the first EOS."
    # So we do a quick trick: temporarily set the first token to a dummy ID (-1)
    # to ensure it won't match 'eos_token_id'. Then revert it.
    batch_size, seq_len = input_ids.shape
    first_tokens = input_ids[:, 0].clone()  # shape: (batch_size,)
    input_ids[:, 0] = -1  # Overwrite with -1 so it can't match eos_token_id

    # Now the "true" earliest EOS is found (beyond the first token).
    eos_1st_positions = (input_ids == eos_token_id).int().argmax(dim=1)  # shape: (batch_size,)

    # Restore the original first tokens.
    input_ids[:, 0] = first_tokens

    # -------------------------------------------------
    # 8. Build a mask to ignore tokens AFTER the first real EOS
    # -------------------------------------------------
    sequence_length = seq_len  # same as input_ids.size(1)
    indices = torch.arange(sequence_length).unsqueeze(0).to(input_ids.device)  # shape: (1, seq_len)

    # For each batch element i, positions <= eos_1st_positions[i] => True,
    # positions > eos_1st_positions[i] => False.
    mask = indices <= eos_1st_positions.unsqueeze(1)  # shape: (batch_size, seq_len)

    # -------------------------------------------------
    # 9. Apply the mask to filter predictions and labels
    # -------------------------------------------------
    filtered_predictions = predicted_token_ids[mask]  # shape: (total_unmasked_tokens,)
    filtered_labels = input_ids[mask]                 # shape: (total_unmasked_tokens,)

    # -------------------------------------------------
    # 10. Compute accuracy
    # -------------------------------------------------
    correct_predictions = (filtered_predictions == filtered_labels).float()
    accuracy = correct_predictions.mean().item()  # scalar float

    return accuracy


def main() -> None:
    """
    Example usage of compute_tfa.
    1) We seed everything for reproducibility.
    2) For each model, we:
        a) Load the model and tokenizer.
        b) Dynamically get bos_token_id, eos_token_id, and pad_token_id from the tokenizer.
        c) Compute TFA on some sample input_texts (and code input for codegemma).
        d) We assert that TFA is in [0,1].
    """
    # 0) Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    # 1) Seed everything
    seed_everything(seed=123)

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
    ]

    # Standard test inputs for TFA
    input_texts = [
        "The happy dog."
        "The quick brown fox jumps over the lazy dog.",
        "Artificial Intelligence is transforming the world of science."
    ]

    # A special code-like string for "google/codegemma-2b"
    code_to_ft = "def solution():\n    return True"
    special_code_input = [
        f"<|fim_prefix|>{code_to_ft}<|fim_suffix|><|fim_middle|>"
    ]

    for config in model_token_configs:
        name = config["name"]
        repo = config["repo"]

        print(f"Evaluating {name} from {repo}")
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

        # Retrieve the actual BOS, EOS, PAD IDs from the tokenizer (if they exist)
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        # 1) TFA on standard input_texts
        tfa_score_general = compute_tfa(
            model=model,
            tokenizer=tokenizer,
            input_texts=input_texts,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id
        )
        # Check TFA is within [0, 1]
        assert 0.0 <= tfa_score_general <= 1.0, (
            f"TFA (General) out of expected [0,1] range: {tfa_score_general}"
        )
        print(f"[{name}] TFA (General Inputs): {tfa_score_general:.4f}")

        # 2) If it's codegemma, also do TFA on the special code input
        if "codegemma-2b" in name.lower():
            tfa_score_code = compute_tfa(
                model=model,
                tokenizer=tokenizer,
                input_texts=special_code_input,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                pad_token_id=pad_id
            )
            # Check TFA is within [0, 1]
            assert 0.0 <= tfa_score_code <= 1.0, (
                f"TFA (Code Input) out of expected [0,1] range: {tfa_score_code}"
            )
            print(f"[{name}] TFA (Special Code Input): {tfa_score_code:.4f}")

        print()  # blank line after each model's results


if __name__ == "__main__":
    main()
