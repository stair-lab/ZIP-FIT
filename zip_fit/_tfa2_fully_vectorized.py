import os
import torch
from typing import List
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)

def teacher_forced_tfa_batch_no_loop(
    prompts: List[str],
    responses: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    max_length: int = 512
) -> List[float]:
    """
    Fully vectorized teacher-forcing accuracy (TFA) for a batch of (prompt, response) pairs,
    with NO per-example loop for the slicing step. Instead, we build index arrays to gather
    all relevant tokens in a single shot.

    Steps:

    1) For i in [0..B), combined_text[i] = prompt[i] + "\n\n" + response[i].
    2) Tokenize all combined_texts => (B, padded_seq_len).
    3) Single forward pass => logits => shape (B, padded_seq_len, vocab_size).
    4) Argmax => preds => shape (B, padded_seq_len).
    5) We also measure each example's prompt length p_len[i] and response length r_len[i]
       by tokenizing them separately with add_special_tokens=False.
    6) Instead of looping i in [0..B), building slices,
       we gather all "teacher-forced" token indices across the entire batch into arrays:

         gather_b[j]      = which example (batch index)
         gather_pred_pos[j]   = position in preds
         gather_label_pos[j]  = position in input_ids (shifted by +1)

       for j in [0.. total_response_tokens-1].

    7) Then we do:
         pred_tokens = preds[gather_b, gather_pred_pos]
         label_tokens = input_ids_batch[gather_b, gather_label_pos]
       => shape: (total_response_tokens,)

       correctness = (pred_tokens == label_tokens).float()

    8) Then we scatter_add or index_add to accumulate correctness per example, and then
       compute correctness_mean[i] = accum[i] / count[i], i in [0..B).

    Returns: List[float] of length B, each is the TFA for example i.

    BOS/EOS/PAD:
      - As always, if you need a pad token, set tokenizer.pad_token_id properly.
      - If your model expects BOS/EOS, either rely on the model's defaults or manually
        insert them in your strings. Just be consistent with how you measure p_len, r_len.

    Shapes:
      - B: number of examples
      - padded_seq_len: the max length after padding
      - total_response_tokens: sum(r_len[i]) for i in [0..B)

    This code is more advanced than the simpler approach that loops over examples to slice.
    Use it if you want single-pass “maximum vectorization” and are comfortable with indexing.
    """

    model.eval()
    model.to(device)

    B = len(prompts)
    assert B == len(responses), "Mismatch: #prompts != #responses"

    # -------------------------------------------------
    # 1) Combine prompts + responses
    # -------------------------------------------------
    combined_texts = [
        prompts[i] + "\n\n" + responses[i]
        for i in range(B)
    ]  # shape: List[str] of length B

    # -------------------------------------------------
    # 2) Tokenize in one batch
    # -------------------------------------------------
    encoded_batch = tokenizer(
        combined_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    # shape: (B, padded_seq_len)
    input_ids_batch = encoded_batch["input_ids"].to(device)
    attention_mask_batch = encoded_batch["attention_mask"].to(device)

    # We'll gather p_len[i], r_len[i] for each example
    prompt_lens = []
    response_lens = []
    for i in range(B):
        # separate tokenization for measuring lengths
        p_enc = tokenizer(prompts[i], add_special_tokens=False)
        r_enc = tokenizer(responses[i], add_special_tokens=False)
        prompt_lens.append(len(p_enc["input_ids"]))
        response_lens.append(len(r_enc["input_ids"]))

    # We'll also measure how many tokens are valid for each example ignoring pad
    seq_lens = attention_mask_batch.sum(dim=1).cpu().tolist()  # shape: (B,)

    # -------------------------------------------------
    # 3) Single forward pass
    # -------------------------------------------------
    with torch.no_grad():
        outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
    # shape: (B, padded_seq_len, vocab_size)
    logits = outputs.logits

    # -------------------------------------------------
    # 4) Argmax => shape: (B, padded_seq_len)
    # -------------------------------------------------
    preds = torch.argmax(logits, dim=-1)

    # -------------------------------------------------
    # 5) Build gather indices for the entire batch
    # -------------------------------------------------
    # We'll create a big index array for "which tokens we gather"
    # total_response_tokens = sum(r_len[i]) across i
    total_response_tokens = sum(response_lens)
    # We'll define:
    #   gather_b[j]        => which example index
    #   gather_pred_pos[j] => the position in preds to gather
    #   gather_label_pos[j] => the position in input_ids_batch to gather
    #   example_idx[j]     => store i so we know which example each token belongs to

    gather_b = torch.empty(total_response_tokens, dtype=torch.long, device=device)
    gather_pred_pos = torch.empty(total_response_tokens, dtype=torch.long, device=device)
    gather_label_pos = torch.empty(total_response_tokens, dtype=torch.long, device=device)
    example_idx = torch.empty(total_response_tokens, dtype=torch.long, device=device)

    offset = 0
    for i in range(B):
        seq_len_i = seq_lens[i]
        p_len_i = prompt_lens[i]
        r_len_i = response_lens[i]
        # If not enough space for the response, skip
        if p_len_i + r_len_i >= seq_len_i or (p_len_i + 1 + r_len_i) > seq_len_i:
            # We'll allocate zero gather points => skip
            continue

        # We'll fill gather arrays for r_len_i tokens
        for t in range(r_len_i):
            gather_b[offset] = i
            gather_pred_pos[offset] = p_len_i + t
            gather_label_pos[offset] = p_len_i + 1 + t
            example_idx[offset] = i
            offset += 1

    # If some examples were out of range, offset < total_response_tokens
    # We'll slice them down to "offset"
    gather_b = gather_b[:offset]
    gather_pred_pos = gather_pred_pos[:offset]
    gather_label_pos = gather_label_pos[:offset]
    example_idx = example_idx[:offset]
    actual_total = offset  # number of valid tokens across the entire batch

    if actual_total == 0:
        # no valid tokens => all TFA=0.0
        return [0.0]*B

    # -------------------------------------------------
    # 6) Gather predicted tokens & label tokens
    # -------------------------------------------------
    # shape: (actual_total,)
    gathered_preds = preds[gather_b, gather_pred_pos]
    gathered_labels = input_ids_batch[gather_b, gather_label_pos]

    # correctness => shape: (actual_total,)
    correctness = (gathered_preds == gathered_labels).float()

    # -------------------------------------------------
    # 7) Compute TFA per example using scatter_add or index_add
    # -------------------------------------------------
    # We'll accumulate correctness and counts for each example in the batch
    accum_correct = torch.zeros(B, dtype=torch.float, device=device)
    accum_count = torch.zeros(B, dtype=torch.float, device=device)

    # Each token belongs to example_idx[j], so we do:
    accum_correct.index_add_(0, example_idx, correctness)
    # Also increment the counts
    ones = torch.ones_like(correctness, dtype=torch.float, device=device)
    accum_count.index_add_(0, example_idx, ones)

    # shape: (B,)
    # TFA_i = accum_correct[i] / accum_count[i] if accum_count[i] != 0
    # We'll do a safe division
    tfa_scores = []
    for i in range(B):
        if accum_count[i] > 0:
            tfa_scores.append((accum_correct[i]/accum_count[i]).item())
        else:
            # no tokens => TFA=0.0
            tfa_scores.append(0.0)

    return tfa_scores

def main():
    """
    Demonstration of the fully vectorized teacher-forcing approach with
    no per-example slicing loop.
    """
    import random
    import numpy as np
    from transformers import set_seed as hf_set_seed

    # 1) Basic seeding
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        hf_set_seed(seed)

    # 2) Example: ProofNet dataset
    ds = load_dataset("hoskinson-center/proofnet", split="validation")
    sub_ds = ds.select(range(4))  # 4 examples

    # Build prompts/responses
    prompts = []
    responses = []
    for example in sub_ds:
        # We'll do a trivial prompt, the "nl_statement" as is
        prompt_text = f"Please translate this to Lean:\n{example['nl_statement']}\n"
        response_text = example["formal_statement"]
        prompts.append(prompt_text)
        responses.append(response_text)

    # 3) Load model/tokenizer
    model_name = "gpt2"  # example
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 4) Fully vectorized TFA with no loop
    scores = teacher_forced_tfa_batch_no_loop(
        prompts=prompts,
        responses=responses,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=256
    )

    # 5) Print results
    for i, score in enumerate(scores):
        print(f"Example {i}: TFA={score:.4f}")
    avg_tfa = sum(scores)/len(scores)
    print(f"Average TFA across {len(scores)} examples = {avg_tfa:.4f}")

if __name__ == "__main__":
    main()
