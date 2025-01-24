# # train/data.py

# import sys
# from typing import Optional
# from itertools import chain
# from torch import nn
# from datasets import load_dataset
# from transformers import AutoTokenizer

# def create_blocks(text_data, tokenizer, block_size):
#     """Create blocks of tokens from text data."""
#     eos_token_id = tokenizer.eos_token_id
#     concatenated_tokens = list(chain(*[tokenizer(text)['input_ids'] + [eos_token_id] for text in text_data]))
    
#     total_length = len(concatenated_tokens)
#     total_length = (total_length // block_size) * block_size
    
#     all_tokens = [concatenated_tokens[i: i + block_size] for i in range(0, total_length, block_size)]
#     return all_tokens

# def load_proofnet(
#     ds_path: str, 
#     tok: Optional[AutoTokenizer], 
#     max_length: int, 
#     config: dict,
#     model: Optional[nn.Module] = None,
#     num_proc: Optional[int] = None, # for .map
#     # num_proc: Optional[int] = os.cpu_count(), # for .map
#     # num_proc: Optional[int] = 10, # for .map
#     shuffle: bool = False, # but slows things down
#     truncation: bool = True, 
#     seed: int = 0,
#     end: Optional[int] = None, 
#     split: Optional[str] = None,
#     batched: bool = True        
# ):
#     print(f'{num_proc=}')
#     end: int = end if end is not None else sys.maxsize
#     ds = load_dataset(ds_path, split=split)
#     ds = ds.shuffle(seed=seed) if shuffle else ds
#     try:
#         ds = ds.select(range(end)) if end is not None else ds
#     except Exception as e:
#         ds = ds.take(end) if end is not None else ds