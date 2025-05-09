from typing import Optional
from datasets import Dataset, load_dataset

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

def prepare_gsm8k_tf_eval_dataset(ds_tf_eval: Dataset) -> Dataset:
    """
    Prepare GSM8K dataset for teacher-forced evaluation.
    """
    return ds_tf_eval.map(
        lambda batch: {
            'prompt': [f"question: {problem}\n" for problem in batch['y_grade_school_math_question']],
            'gold_response': [f"answer: {solution}\n### {final_answer}\n" for solution, final_answer in zip(batch['y_grade_school_math_solution'], batch['y_grade_school_math_final_answer'])]
        },
        batched=True,
        num_proc=48
    )

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

def get_eval_tf_datasets(config: dict = {}) -> Dataset:
    """
    Get evaluation datasets for teacher-forced evaluation.
    """
    training_eval_tf_dataset_name = config.get("training_eval_tf_dataset_name", "zipfit/math-select-06062025")
    training_eval_tf_split = config.get("training_eval_tf_split", "train")
    max_eval_tf_samples: Optional[int] = config.get('max_eval_tf_samples', None)
    split_str = f'{training_eval_tf_split}[:{max_eval_tf_samples}]' if max_eval_tf_samples else training_eval_tf_split

    if 'Putnam-AXIOM-for-zip-fit-splits' in training_eval_tf_dataset_name:
        ds_tf_eval = load_dataset(training_eval_tf_dataset_name, split=split_str).with_format('torch')
        # 1st Stage: Prepare strings
        ds_tf_eval = prepare_putnam_tf_eval_dataset(ds_tf_eval)

        # 2nd Stage: Tokenize and group text into blocks
        pass # Due to how TF evals work tokenization is done in the tf function.
    elif 'gsm8k' in training_eval_tf_dataset_name:
        ds_tf_eval = load_dataset(training_eval_tf_dataset_name, split=split_str).with_format('torch')
        # 1st Stage: Prepare strings
        ds_tf_eval = prepare_gsm8k_tf_eval_dataset(ds_tf_eval)

        # 2nd Stage: Tokenize and group text into blocks
        pass # Due to how TF evals work tokenization is done in the tf function.    
    elif 'UDACA/proofnet-v3-lean4' in training_eval_tf_dataset_name:
        ds_tf_eval = load_dataset(training_eval_tf_dataset_name, split=split_str).with_format('torch')
        # 1st Stage: Prepare strings
        ds_tf_eval = prepare_proofnet_tf_eval_dataset(ds_tf_eval)

        # 2nd Stage: Tokenize and group text into blocks
        pass # Due to how TF evals work tokenization is done in the tf function.
    else:
        raise ValueError(f"Unsupported dataset name: {training_eval_tf_dataset_name}")

    return ds_tf_eval