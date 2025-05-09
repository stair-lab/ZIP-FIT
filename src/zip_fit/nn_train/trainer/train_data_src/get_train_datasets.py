from datasets import Dataset, load_dataset
from typing import Optional

from zip_fit.nn_train.trainer.prompts.train_math_prompt_templates import format_gsm8k_prompt
from zip_fit.nn_train.trainer.train_data_src.prepare_train_tokenization import tokenize_and_group_texts_via_blocks

def get_train_datasets(tokenizer, config: dict = {}) -> Dataset:
    """ Get the training dataset. """
    training_dataset_name = config.get("training_dataset_name", "zipfit/math-select-06062025")
    training_split = config.get("training_split", "train")
    block_size: int = config.get('block_size', 1024)
    max_samples: Optional[int] = config.get('max_train_samples', None)
    split_str = f'{training_split}[:{max_samples}]' if max_samples else training_split
    
    # Load datasets based on name
    if training_dataset_name.startswith("zipfit/math-select"):
        ds_train = load_dataset(training_dataset_name, split=split_str).with_format('torch')

        # 1st Stage: Prepare strings
        pass # Already prepared, according to this: /home/brandomiranda/ZIP-FIT/src/zip_fit/nn_train/trainer/prompts/train_math_prompt_templates.py
        
        # 2nd Stage: Prepare dataset for training
        ds_train = ds_train.map(
            # Note: data set can be used directly as is, no need to process the text.
            # Note: tokenization happens in the tokenize_and_group_texts_via_blocks function.
            lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
            batched=True,
            remove_columns=[col for col in ds_train.column_names if col != 'text'],
            num_proc=48,
        ) 
    elif 'gsm8k' in training_dataset_name:
        ds_train = load_dataset(training_dataset_name, split=split_str).with_format('torch')

        # 1st Stage: Prepare strings
        ds_train = ds_train.map(
            lambda batch: {"text": [format_gsm8k_prompt(q, a, fa) for q, a, fa in zip(
                batch["y_grade_school_math_question"], 
                batch["y_grade_school_math_solution"], 
                batch["y_grade_school_math_final_answer"]
            )]},
            batched=True,
            remove_columns=ds_train.column_names,
            num_proc=48,
        )

        # 2nd Stage: tokenize and group text into blocks
        ds_train = ds_train.map(
            lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
            batched=True,
            remove_columns=ds_train.column_names,
            num_proc=48,
        )
    elif training_dataset_name == "UDACA/proofnet-v3-lean4":
        ds_train = load_dataset(training_dataset_name, split=split_str).with_format('torch')

        # 1st Stage: Prepare strings TODO: if you actually decided to to Autoformalization check this train formatting and think thorugh your eval to decide what to do.
        ds_train = ds_train.map(
            lambda batch: {"text": [
                # format_lean4_prompt(nl_stmt) + " " + header + "\n\n" + formal_stmt 
                f'informal statement: {nl_stmt}\n\nformal statement: {header}\n\n{formal_stmt}\n\n'
                for nl_stmt, header, formal_stmt in zip(
                    # batch["informal_prefix"], # this basically pre-trains on the code
                    batch["nl_statement"], # this is more of a "Informal {nl_stmt}: Formal {formal_stmt}" training format but the formatting would have to chagne for this to be the case. 
                    batch["header_no_import"],
                    batch["formal_statement"]
                )
            ]},
            batched=True,
            remove_columns=ds_train.column_names,
            num_proc=48,
        )

        # 2nd Stage: tokenize and group text into blocks
        ds_train = ds_train.map(
            lambda batch: tokenize_and_group_texts_via_blocks(batch, tokenizer=tokenizer, block_size=block_size),
            batched=True,
            remove_columns=ds_train.column_names,
            num_proc=48,
        )

    else:
        raise ValueError(f"Unsupported dataset name: {training_dataset_name}")

    return ds_train

