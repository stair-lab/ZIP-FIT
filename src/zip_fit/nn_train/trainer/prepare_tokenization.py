"""
Functions for tokenizing and preparing text for training language models.
This module handles the tokenization and grouping of text into blocks.
"""

import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer
from itertools import chain

def tokenize_and_group_texts_via_blocks(
    examples: Dict[str, List[str]],  # since Batched=True gives a list of strings, note: a block size will be 1 sequences, a concat of tokenized rows from the data set! 
    tokenizer: AutoTokenizer,
    block_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a batch of raw text examples and groups the tokens into fixed-size blocks.

    This function is designed for use with Hugging Face datasets (with batched=True). It processes
    a batch of examples (where each example is a string under the 'text' key) by performing the following steps:

      1. **Retrieve Special Token IDs:**  
         It retrieves the beginning-of-sequence (BOS) token ID and the end-of-sequence (EOS) token ID from the tokenizer.
         - The BOS token ID is obtained from `tokenizer.bos_token_id` (if available).
         - The EOS token ID is obtained from `tokenizer.eos_token_id`.

      2. **Per-Text Tokenization with Special Tokens:**  
         For each text in the input list:
           - It prepends the BOS token (if available).
           - Tokenizes the text using the tokenizer (without adding any special tokens automatically).
           - Appends the EOS token to the token list.
         This is done individually for each text so that we can explicitly control the addition of special tokens.

      3. **Concatenation:**  
         All token lists are concatenated into one long list of tokens. This is achieved using the `chain` function
         to flatten the list of lists.

      4. **Truncation:**  
         The total token list is truncated so that its length is an exact multiple of `block_size`.  
         This is done by discarding any tokens at the end that would not complete a full block.

      5. **Blocking:**  
         The truncated token list is split into contiguous blocks, each of length `block_size`.

      6. **Label Creation:**  
         For language modeling tasks, the function creates a 'labels' field that is an exact copy of the 'input_ids'
         blocks. Both fields are converted into PyTorch tensors with dtype torch.long.

    **Note on the attention mask field:**  
    The returned dictionary does not include an `attention_mask` field. Including an attention mask with a value of `None` 
    may cause issues with the Hugging Face Trainer, which expects either a valid tensor or for the key to be omitted entirely.

    Args:
        examples (Dict[str, List[str]]):
            A dictionary representing a batch of examples from the dataset. It must contain a key 'text'
            whose value is a list of strings (one per example).
        tokenizer (AutoTokenizer):
            A tokenizer instance (e.g., from Hugging Face's transformers) that can process a single text string.
            The tokenizer should return a dictionary containing at least the key 'input_ids' when called with a text,
            and it must have attributes `bos_token_id` (which may be None) and `eos_token_id`.
        block_size (int):
            The desired number of tokens per block (for example, 1024 or 4096).

    Returns:
        Dict[str, torch.Tensor]:
            A dictionary where:
              - 'input_ids' maps to a PyTorch tensor of token blocks, each block being a list of token IDs of length `block_size`
                and of dtype torch.long.
              - 'labels' maps to a copy of the 'input_ids' tensor.
            
            **Note:** No `attention_mask` is included in the returned dictionary.
    """
    # -------------------------------------------------------------------------
    # Step 1: Retrieve Special Token IDs.
    # Retrieve the end-of-sequence (EOS) token ID from the tokenizer.
    eos_token_id: int = tokenizer.eos_token_id
    # Retrieve the beginning-of-sequence (BOS) token ID from the tokenizer.
    # If not used, bos_token_id will be None.
    bos_token_id: Optional[int] = tokenizer.bos_token_id
    # pad_token: Optional[int] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    # -------------------------------------------------------------------------
    # Step 2: Per-Text Tokenization with Special Tokens.
    # For each text in the input list, perform the following:
    #   - Prepend the BOS token if it exists.
    #   - Tokenize the text to get a dictionary; extract the token IDs from 'input_ids'.
    #   - Append the EOS token to the token list.
    # The list comprehension processes each text individually.
    token_lists: List[List[int]] = [
        # If a BOS token exists, prepend it; otherwise, use an empty list.
        ([bos_token_id] if bos_token_id is not None else []) +
        tokenizer(text)['input_ids'] + [eos_token_id]  # Tokenize text and append EOS token.
        for text in examples["text"]
    ]

    # -------------------------------------------------------------------------
    # Step 3: Concatenate tokenized outputs across the batch.
    # Flatten the list of token lists into a single long list using chain.
    concatenated_tokens: List[int] = list(chain(*token_lists))

    # -------------------------------------------------------------------------
    # Step 4: Compute the total length and adjust to an exact multiple of block_size.
    total_length: int = len(concatenated_tokens)
    # Calculate the number of complete blocks that can be formed.
    num_blocks: int = total_length // block_size
    # Adjust total_length to discard any extra tokens that do not form a complete block.
    total_length = num_blocks * block_size
    
    # -------------------------------------------------------------------------
    # Step 5: Split the concatenated token list into blocks of fixed size.
    # The list comprehension iterates over the concatenated token list in steps of block_size,
    # slicing out a block each time.
    all_token_blocks: List[List[int]] = [
        concatenated_tokens[i: i + block_size] for i in range(0, total_length, block_size)
    ]

    # -------------------------------------------------------------------------
    # Step 6: Create labels for language modeling tasks.
    # It is common for language models to use the input_ids as labels.
    # Convert the list of token blocks into PyTorch tensors with dtype=torch.long.
    result: Dict[str, torch.Tensor] = {
        "input_ids": torch.tensor(all_token_blocks, dtype=torch.long),
        "labels": torch.tensor(all_token_blocks.copy(), dtype=torch.long),
    }

    return result 