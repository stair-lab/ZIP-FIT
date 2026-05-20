import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedModel

def tfce_loss_full_gold_ref(
    prompt: str,
    gold_response: str,
    model: PreTrainedModel,
    repo: str,
    device: str = "cuda",
    reduction: str = "mean"
) -> float:
    """
    Teacher-forced cross-entropy (TFCE), ignoring the prompt portion.
    
    => -log p(gold_response | prompt), but only computing CE over the gold_response tokens.

    Steps:
      1) combined_text = prompt + "\n\n" + gold_response
      2) Tokenize => shape (1, seq_len) with add_special_tokens=False
      3) Forward pass => logits shape (1, seq_len, vocab_size)
      4) Re-tokenize prompt/gold_response to measure lengths (len_prompt, len_gold)
      5) Build labels from input_ids => set label = -100 for [0..len_prompt-1] and beyond [len_prompt+len_gold-1]
      6) Flatten => cross_entropy(..., ignore_index=-100, reduction=reduction)
      7) Return float scalar (mean or sum over gold_response tokens).

    If text is truncated or too short, we return 0.0 as fallback.

    We do not forcibly add BOS/EOS, matching a "bare-bones" style (like your TFA code).
    """

    # 1) Load the tokenizer from the same repo
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # 2) Combine text
    combined_text = prompt + "\n\n" + gold_response

    # 3) Tokenize => shape (1, seq_len)
    enc = tokenizer(
        combined_text,
        return_tensors="pt",
        add_special_tokens=False
    )
    input_ids = enc["input_ids"].to(device)  # shape => (1, seq_len)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)  # shape => (1, seq_len)

    # measure lengths
    batch_size, seq_len = input_ids.shape

    # Re-tokenize prompt / gold_response to measure how many tokens each has
    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    len_prompt = len(prompt_enc["input_ids"])

    gold_enc = tokenizer(gold_response, add_special_tokens=False)
    len_gold = len(gold_enc["input_ids"])

    # If truncated or too short
    if (len_prompt + len_gold) > seq_len:
        return 0.0

    # 4) forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # shape => (1, seq_len, vocab_size)
    logits = outputs.logits
    vocab_size = logits.size(-1)

    # 5) Build labels from input_ids
    labels = input_ids.clone()  # (1, seq_len)
    # ignore the prompt portion
    labels[0, :len_prompt] = -100
    # ignore anything beyond gold_response portion
    end_idx = len_prompt + len_gold
    if end_idx < seq_len:
        labels[0, end_idx:] = -100

    # 6) Flatten
    logits_2d = logits.view(-1, vocab_size)  # (seq_len, vocab_size)
    labels_1d = labels.view(-1)             # (seq_len,)

    # 7) cross_entropy => ignoring -100
    ce_loss = F.cross_entropy(
        logits_2d,
        labels_1d,
        ignore_index=-100,
        reduction=reduction
    )
    return float(ce_loss.item())


def compute_tfce_for_subds(
    sub_ds: Dataset,
    model: PreTrainedModel,
    repo: str,
    device: str = "cuda",
    reduction: str = "mean",
    debug: bool = False
) -> float:
    """
    Process an entire subset of data (sub_ds) and compute the average TFCE across all examples.

    For each example, we do `tfce_loss_full_gold_ref(...)`. We average over examples.

    sub_ds[i] must have 'prompt' and 'gold_response'.
    If `debug=True`, we print each example's cross-entropy.
    """
    sum_loss = 0.0
    count = 0

    for i, example in enumerate(sub_ds):
        prompt = example["prompt"]
        gold_response = example["gold_response"]

        loss_i = tfce_loss_full_gold_ref(
            prompt=prompt,
            gold_response=gold_response,
            model=model,
            repo=repo,
            device=device,
            reduction=reduction
        )
        sum_loss += loss_i
        count += 1

        if debug:
            print(f" Example {i}: TFCE loss={loss_i:.4f}")

    return sum_loss / count if count > 0 else 0.0
