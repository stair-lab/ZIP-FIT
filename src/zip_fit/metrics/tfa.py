import torch
from transformers import AutoTokenizer, PreTrainedModel
from datasets import Dataset

def tfa_teacher_forced_accuracy(
    prompt: str,
    gold_response: str,
    model: PreTrainedModel,
    repo: str, # HF repo name needed to load the correct tokenizer
    device: str = "cuda"
) -> float:
    """
    Teacher-forced accuracy (token-level) on `gold_response` given a concatenated text = prompt + gold_response.

    Steps:
      1) Combined text = prompt + "\n\n" + gold_response
      2) Tokenize combined text => shape: (1, total_seq_len)
      3) Forward pass => logits shape: (1, total_seq_len, vocab_size)
      4) Identify the token range for the gold_response
      5) Compare the predicted tokens in that range with the reference gold_response tokens
      6) Return fraction matched in [0, 1]

    Notes about BOS/EOS/PAD:
      - Because we do per-example calls (prompt+gold_response) only, no extra padding is needed.
      - We do not forcibly add BOS or EOS here. We skip it to match a "bare-bones" style,
        similar to the updated tfa.py that also ignores explicit BOS/EOS tokens.
      - If the combined text is truncated or too short, we return 0.0 as a fallback.

    Correctness:
      - We allow the to
    """
    # 1) Combine text
    combined_text = prompt + "\n\n" + gold_response

    # 2) Get tokenizer from the `repo` -- getting from repo to ensure user doesn't give the wrong tokenizer TODO verify this statement with more sanity checks
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # 3) Tokenize entire reference
    enc = tokenizer(combined_text, return_tensors="pt")
    # shape: (1, total_seq_len)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape: (1, total_seq_len, vocab_size)
    preds = torch.argmax(logits, dim=-1)  # shape: (1, total_seq_len)

    # 4) Tokenize the gold_response alone to find how many tokens it has
    gold_response_enc = tokenizer(gold_response, add_special_tokens=False)
    len_gold_response = len(gold_response_enc["input_ids"])

    # Tokenize the prompt alone for length
    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    len_prompt = len(prompt_enc["input_ids"])

    total_seq_len = input_ids.size(1)

    # If the combined text is too short or the gold_response doesn't fit, skip
    if len_prompt + len_gold_response >= total_seq_len:
        return 0.0

    # Teacher forcing alignment:
    #   model's position t attempts to predict token at position t+1
    pred_slice = preds[:, len_prompt : (len_prompt + len_gold_response)]
    label_slice = input_ids[:, (len_prompt + 1) : (len_prompt + 1 + len_gold_response)]

    if pred_slice.size(1) == 0 or label_slice.size(1) == 0:
        return 0.0

    correctness = (pred_slice == label_slice).float()  # shape: (1, number_of_gold_response_tokens)
    acc = correctness.mean().item()
    return acc


def compute_tfa_for_subds(
    sub_ds: Dataset,
    model: PreTrainedModel,
    repo: str, # HF repo name needed to load the correct tokenizer
    device: str = "cuda",
    debug: bool = False,
) -> float:
    """
    Process an entire subset of data (sub_ds) and compute the average TFA across all examples.

    Parameters:
      sub_ds: The subset of the dataset (like a HuggingFace 'Dataset' slice).
      model:  A language model (transformers PreTrainedModel).
      repo:   The model repo string, used to load the correct tokenizer in tfa_teacher_forced_accuracy.
      device: 'cuda' or 'cpu'.

    Returns:
      float: The average TFA over all examples in sub_ds.
    """
    sum_acc = 0.0
    count = 0

    for i, example in enumerate(sub_ds):
        prompt = example["prompt"]
        gold_response = example["gold_response"]

        acc_i = tfa_teacher_forced_accuracy(
            prompt=prompt,
            gold_response=gold_response,
            model=model,
            repo=repo,
            device=device
        )
        sum_acc += acc_i
        count += 1

        print(f" Example {i}: TFA = {acc_i:.4f}") if debug else None

    return sum_acc / count if count > 0 else 0.0
