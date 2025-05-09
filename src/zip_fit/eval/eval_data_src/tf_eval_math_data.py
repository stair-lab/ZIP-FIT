from typing import Dict, Tuple, Any, Optional, List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from zip_fit.nn_train.trainer.train_data_src.prepare_tokenization import tokenize_and_group_texts_via_blocks


def prepare_putnam_tf_eval_dataset(ds_tf_eval: Dataset) -> Dataset:
    """
    Prepare Putnam dataset for teacher-forced evaluation.
    
    Args:
        ds_tf_eval: Raw evaluation dataset
    
    Returns:
        Dataset: Processed dataset ready for teacher-forced evaluation
    """
    return ds_tf_eval.map(
        lambda batch: {
            'prompt': [format_putnam_eval_prompt(p) for p in batch['problem']],
            'gold_response': [solution for solution in batch['solution']]
        },
        batched=True,
        num_proc=48
    )

def prepare_math_tf_datasets(
    tokenizer: AutoTokenizer,
    config: Dict[str, Any] = {},
    dataset_name: str = "putnam"
) -> Tuple[Dataset, Dataset]:
    """
    Prepare math datasets for evaluation.
    
    Args:
        tokenizer: Tokenizer to use for tokenization
        config: Configuration dictionary with optional parameters
        dataset_name: Type of dataset to use for evaluation ("gsm8k", "putnam")
    
    Returns:
        Tuple[Dataset, Dataset]: Evaluation dataset and teacher-forced evaluation dataset
    """
    block_size: int = config.get('block_size', 1024)
    print(f'{block_size=}')
    
    # Load datasets based on name
    if dataset_name == "putnam":
        # Get dataset parameters from config
        hf_dataset_name = config.get('dataset_name', "zipfit/Putnam-AXIOM-for-zip-fit-splits")
        split = config.get('split', "test")
        
        # Load Putnam dataset for evaluation
        raw_eval_dataset = load_dataset(hf_dataset_name, split=split).with_format('torch')
        raw_tf_eval_dataset = load_dataset(hf_dataset_name, split=split).with_format('torch')
        
        # Prepare datasets for evaluation
        eval_dataset = prepare_putnam_eval_dataset(raw_eval_dataset, tokenizer, block_size)
        tf_eval_dataset = prepare_putnam_tf_eval_dataset(raw_tf_eval_dataset)
    
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    
    # Print dataset information
    print(f'{len(eval_dataset)=} {eval_dataset.column_names=}')
    print(f'{len(tf_eval_dataset)=} {tf_eval_dataset.column_names=}')
    
    return eval_dataset, tf_eval_dataset
