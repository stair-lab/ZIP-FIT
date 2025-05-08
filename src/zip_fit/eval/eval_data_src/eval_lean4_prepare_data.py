from typing import List, Tuple

from zip_fit.eval.prompts.eval_lean4_prompt_temmpaltes import my_prompt_format

from datasets import Dataset

def prepare_eval_data(config: dict = {}) -> Tuple[List[str], List[str], Dataset]:
    from datasets import load_dataset
    ds_test = load_dataset('UDACA/proofnet-v3-lean4', split='test')
    ds_test = ds_test.select(list(range(2)))  # optionally select a subset
    prompts = [my_prompt_format(row['nl_statement']) for row in ds_test]
    gold_headers = [row['header_no_import'] for row in ds_test]
    return prompts, gold_headers, ds_test
