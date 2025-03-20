"""
Evaluate cross-entropy loss on the 'AI4M/gemma2-2b-gpt4-more-5-epochs' model using the
'HuggingFace Trainer' and the 'UDACA/proofnet-v3-lean4' dataset, ensuring the same
"block packing" approach that was used during training.

We'll use a custom tokenization/grouping step to create fixed-length blocks of tokens
from the raw text, just as in a typical language-model training loop. Then we rely on
the Hugging Face Trainer for the evaluation step, which conveniently computes the
average cross-entropy (and logs it as `'eval_loss'`).

By default, HF's Trainer returns the "average token-level negative log-likelihood"
in `'eval_loss'`. You can then take `math.exp(eval_loss)` to see the equivalent
perplexity if desired.

Requirements:
    python>=3.8
    torch
    transformers
    datasets
    nltk
    accelerate (if using multi-GPU or a distributed setup)

Usage:
    python evaluate_cross_entropy_trainer.py
"""

import torch
from itertools import chain
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

###############################################################################
# 2. Block-packing function
###############################################################################

def tokenize_and_group_texts_via_blocks(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    block_size: int = 1024,
) -> Dict[str, Any]:
    """
    Tokenizes raw text examples and groups tokens into fixed-size blocks, 
    replicating the typical approach for causal language-modeling training.

    Steps:
      1) For each string in 'examples["text"]':
         - Optionally prepend BOS if tokenizer.bos_token_id is not None
         - Tokenize with add_special_tokens=False
         - Append EOS if tokenizer.eos_token_id is not None
      2) Concatenate all token IDs into a single long list.
      3) Truncate so that the final length is multiple of block_size.
      4) Break into contiguous segments (blocks) of block_size tokens.
      5) Return {'input_ids': blocks, 'labels': blocks} as lists of lists.

    We'll rely on the Trainer's default collator or a simple data collator
    that can convert these to torch tensors at batch time.

    Args:
        examples (Dict[str, List[str]]): A batch of examples containing a "text" key.
        tokenizer (AutoTokenizer): A Hugging Face tokenizer, e.g. GPT2Tokenizer.
        block_size (int): The desired length of each token block.

    Returns:
        Dict[str, List[List[int]]]:
            A dictionary with keys 'input_ids' and 'labels', each a list of 
            integer token blocks of length 'block_size'.
    """
    bos_token_id: Optional[int] = tokenizer.bos_token_id
    eos_token_id: Optional[int] = tokenizer.eos_token_id

    # 1. Tokenize each example separately, including BOS/EOS if available.
    tokenized_texts = []
    for txt in examples["text"]:
        tokens = []
        if bos_token_id is not None:
            tokens.append(bos_token_id)

        # Tokenize
        res = tokenizer(txt, add_special_tokens=False)
        tokens.extend(res["input_ids"])

        if eos_token_id is not None:
            tokens.append(eos_token_id)

        tokenized_texts.append(tokens)

    # 2. Flatten all tokenized outputs
    concatenated = list(chain(*tokenized_texts))

    # 3. Truncate to a multiple of block_size
    total_length = (len(concatenated) // block_size) * block_size
    concatenated = concatenated[:total_length]

    # 4. Break into blocks
    result_input_ids = [
        concatenated[i : i + block_size]
        for i in range(0, total_length, block_size)
    ]

    # 5. For LM, 'labels' == 'input_ids'
    return {
        "input_ids": result_input_ids,
        "labels": [block.copy() for block in result_input_ids],  # avoid same list reference
    }


###############################################################################
# 3. Hugging Face Dataset -> Flatten blocks into a single item per block
###############################################################################

class BlocksAsExamplesDataset(TorchDataset):
    """
    A small utility dataset class that flattens the "list of blocks" returned by
    the map function into separate examples. If a row in the dataset has N blocks,
    we yield N separate examples. This allows the HF Trainer to see each block 
    as an individual sample.

    The dataset is expected to have columns: 'input_ids', 'labels' 
    which are lists of integer lists. E.g. shape (Nblocks, block_size).

    Example usage:
        raw_dataset = raw_dataset.map(tokenize_and_group_texts_via_blocks, batched=True)
        # 'raw_dataset' now has columns with lists-of-lists.
        eval_dataset = BlocksAsExamplesDataset(raw_dataset)
    """
    def __init__(self, dataset: Dataset):
        self.inner_dataset = dataset
        self.offsets = []  # (index_in_inner_dataset, index_in_that_row_block)
        cum_count = 0

        # We read each row. If row["input_ids"] is a list of B blocks, 
        # then we add B offsets pointing to that row, block index.
        for row_idx in range(len(dataset)):
            row = dataset[row_idx]
            blocks_count = len(row["input_ids"])  # number of blocks in that row
            for i in range(blocks_count):
                self.offsets.append((row_idx, i))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_idx, block_idx = self.offsets[idx]
        row = self.inner_dataset[row_idx]
        # Convert the single block (list of ints) into a torch tensor
        input_ids_block = row["input_ids"][block_idx]
        labels_block = row["labels"][block_idx]

        return {
            "input_ids": torch.tensor(input_ids_block, dtype=torch.long),
            "labels": torch.tensor(labels_block, dtype=torch.long),
        }
 