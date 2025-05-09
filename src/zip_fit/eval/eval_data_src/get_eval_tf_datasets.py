from datasets import Dataset
from zip_fit.eval.eval_data_src.eval_lean4_prepare_data import format_lean4_prompt

def prepare_proofnet_tf_eval_dataset(ds_tf_eval: Dataset) -> Dataset:
    """
    Prepare ProofNet dataset for teacher-forced evaluation.
    """
    return ds_tf_eval.map(
        lambda batch: {
            'prompt': [f'informal statement: {nl_stmt}\n\n' for nl_stmt in batch['nl_statement']],
            'gold_response': [f"formal statement: {header}\n\n{formal_stmt}\n\n" 
                              for header, formal_stmt in zip(batch['header_no_import'], batch['formal_statement'])]
        },
        batched=True,
        num_proc=48
    )

def prepare_putnam_tf_eval_dataset(ds_tf_eval: Dataset) -> Dataset:
    """
    Prepare Putnam dataset for teacher-forced evaluation.
    """
    return ds_tf_eval.map(
        lambda batch: {
            'prompt': [f"Problem: {problem}\n\n" for problem in batch['problem']],
            'gold_response': [f"Solution: {solution}\n\n" for solution in batch['solution']]
        },
        batched=True,
        num_proc=48
    )

def get_eval_tf_datasets(config: dict = {}) -> Dataset:
    """
    Get evaluation datasets for teacher-forced evaluation.
    """
    eval_tf_ataset_name = config.get("eval_dataset_name", "zipfit/math-select-06062025")
    eval_split = config.get("eval_split", "train")
    split_str = f'{eval_split}[:{max_samples}]' if max_samples else eval_split

    if  