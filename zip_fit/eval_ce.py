#!/usr/bin/env python
"""
Evaluation Script for AI4M/gemma2-2b-gpt4-more-5-epochs on UDACA/proofnet-v3-lean4 Dataset.

This script:
  1. Loads the model "AI4M/gemma2-2b-gpt4-more-5-epochs" and its tokenizer.
  2. Loads the full "validation" split of the dataset "UDACA/proofnet-v3-lean4".
  3. Creates a "text" field for each example (using "header_no_import" if available,
     otherwise "header") to mimic the training setup.
  4. Tokenizes and groups texts into fixed-size blocks in the same manner as during training.
  5. Uses the Hugging Face Trainer to evaluate (without training) on the entire dataset,
     ensuring that no evaluation steps are skipped.
  6. Provides a function to generate text completions using the model's generation method.
  7. Prints the computed cross-entropy loss.

Key points:
  - No training parameters like logging_steps are used since we are not training.
  - No max_eval_steps is set so that Trainer.evaluate() iterates over the entire eval_dataset.
  - The tokenization and grouping function is identical to that used in training, ensuring consistency.
  
Usage:
    pip install transformers datasets torch
    python eval_loss.py
"""

import torch
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset


def tokenize_and_group_texts_via_blocks(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    block_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a batch of raw text examples and groups the tokens into fixed-size blocks.

    This replicates the training-time processing by:
      1. Tokenizing each text individually and adding BOS/EOS tokens.
      2. Concatenating all tokenized outputs into a single long list.
      3. Truncating to an exact multiple of block_size.
      4. Splitting the list into blocks.
      5. Creating labels identical to the input_ids.

    Args:
        examples (Dict[str, List[str]]): A dict with key "text" mapping to a list of strings.
        tokenizer (AutoTokenizer): The tokenizer instance.
        block_size (int): Fixed block length (e.g. 1024).

    Returns:
        Dict[str, torch.Tensor]: A dict with "input_ids" and "labels" tensors.
    """
    from itertools import chain

    # Retrieve special token IDs.
    eos_token_id: int = tokenizer.eos_token_id
    bos_token_id: Optional[int] = tokenizer.bos_token_id

    # Tokenize each text, prepending BOS (if available) and appending EOS.
    token_lists: List[List[int]] = [
        (([bos_token_id] if bos_token_id is not None else []) +
         tokenizer(text)["input_ids"] + [eos_token_id])
        for text in examples["text"]
    ]

    # Flatten the list of token lists.
    concatenated_tokens: List[int] = list(chain(*token_lists))

    # Truncate to a length that is an exact multiple of block_size.
    total_length: int = len(concatenated_tokens)
    num_blocks: int = total_length // block_size
    total_length = num_blocks * block_size

    # Split into blocks.
    all_token_blocks: List[List[int]] = [
        concatenated_tokens[i: i + block_size] for i in range(0, total_length, block_size)
    ]

    input_ids = torch.tensor(all_token_blocks, dtype=torch.long)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def generate_completion(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_length: int = 512) -> str:
    """
    Generates a text continuation from a given prompt using the model.

    Uses greedy decoding (do_sample=False) to generate up to max_length tokens.

    Args:
        prompt (str): The text prompt.
        model (AutoModelForCausalLM): The causal language model.
        tokenizer (AutoTokenizer): The corresponding tokenizer.
        max_length (int, optional): Maximum length (default is 512).

    Returns:
        str: The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    output_ids = model.generate(input_ids, max_length=max_length, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    """
    Main function to evaluate the model.

    Steps:
      1. Load the model and tokenizer.
      2. Load the full validation dataset from UDACA/proofnet-v3-lean4.
      3. Map each example to create a "text" field (from "header_no_import" or "header").
      4. Tokenize and group the texts into fixed-size blocks using the training function.
      5. Configure TrainingArguments for evaluation only—ensuring the full dataset is processed.
      6. Initialize the HF Trainer and call evaluate() to compute the cross-entropy loss.
      7. Print the evaluation loss.
    """
    # Model and dataset identifiers.
    model_name: str = "AI4M/gemma2-2b-gpt4-more-5-epochs"
    dataset_name: str = "UDACA/proofnet-v3-lean4"
    eval_split: str = "validation"  # Use full validation set.
    block_size: int = 1024         # Must match training configuration.

    # Load model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load the evaluation dataset.
    dataset = load_dataset(dataset_name, split=eval_split)

    # Create a "text" field from "header_no_import" (fallback to "header").
    def select_text(example: Dict[str, Any]) -> Dict[str, str]:
        text = example.get("header_no_import") or example.get("header")
        return {"text": text} if text is not None else {}

    dataset = dataset.map(select_text, remove_columns=[col for col in dataset.column_names if col != "text"])
    dataset = dataset.filter(lambda example: example["text"] is not None)

    # Tokenize and group texts into blocks.
    dataset = dataset.map(
        lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "labels"])

    # Set up TrainingArguments for evaluation only.
    # Note: We do not use logging_steps here since we are not training.
    training_args = TrainingArguments(
        output_dir="./eval_output",         # Required output directory.
        per_device_eval_batch_size=1,         # Batch size for evaluation.
        do_eval=True,                         # Enable evaluation mode.
        report_to=["none"],                   # Disable external logging.
        # No max_eval_steps is specified; Trainer.evaluate() processes the full dataset.
    )

    # Initialize the HF Trainer with the full eval_dataset.
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,   # Full validation set.
        tokenizer=tokenizer,
    )

    # Evaluate the model over the entire dataset.
    eval_result: Dict[str, float] = trainer.evaluate()
    eval_loss: float = eval_result.get("eval_loss", float("nan"))
    print(f"Evaluation Cross-Entropy Loss: {eval_loss:.4f}")

    # Example usage of the generation function:
    example_prompt = dataset[0]["input_ids"]
    # (For generation, decode the input_ids back to text)
    prompt_text = tokenizer.decode(example_prompt, skip_special_tokens=True)
    generated_text = generate_completion(prompt_text, model, tokenizer, max_length=512)
    print("\nSample Prompt:")
    print(prompt_text)
    print("\nGenerated Completion:")
    print(generated_text)

if __name__ == "__main__":
    main()
