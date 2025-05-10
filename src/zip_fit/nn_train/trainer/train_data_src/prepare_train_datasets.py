from datasets import Dataset, load_dataset
from typing import Optional

from zip_fit.nn_train.trainer.prompts.train_math_prompt_templates import format_gsm8k_prompt_to_q_a_fa, format_zipfit_math_select_prompt_to_prob_soln
from zip_fit.nn_train.trainer.train_data_src.prepare_train_tokenization import tokenize_and_group_texts_via_blocks

def get_train_datasets(
        tokenizer, 
        dataset_name: str,
        split: Optional[str] = None, 
        config: dict = {}
        ) -> Dataset:
    """ Get the training dataset. """
    block_size: int = config.get('block_size', 1024)
    max_samples: Optional[int] = config.get('max_train_samples', None)
    training_split: str = config.get("training_split", "train") if split is None else split # might override split if you want the data in the train format for evaluating CE during training.
    split_str: str = f'{training_split}[:{max_samples}]' if max_samples else training_split
    
    # Load datasets based on name
    if dataset_name.startswith("zipfit/math-select"):
        ds_train = load_dataset(dataset_name, split=split_str).with_format('torch')
        # One-pass tokenise-and-group
        ds_train = ds_train.map(
            lambda batch: tokenize_and_group_texts_via_blocks(
                batch, tokenizer=tokenizer, block_size=1024
            ),
            batched=True,
            remove_columns=ds_train.column_names,   # drop *all* original columns, incl. "text"
            num_proc=48,
        )
    elif 'gsm8k' in dataset_name:
        ds_train = load_dataset(dataset_name, split=split_str).with_format('torch')

        # 1st Stage: Prepare strings
        ds_train = ds_train.map(
            lambda batch: {"text": [format_gsm8k_prompt_to_q_a_fa(q, a, fa) for q, a, fa in zip(
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
    elif dataset_name == "UDACA/proofnet-v3-lean4":
        ds_train = load_dataset(dataset_name, split=split_str).with_format('torch')

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

    elif 'Putnam-AXIOM' in dataset_name:
        ds_train = load_dataset(dataset_name, split=split_str).with_format('torch')

        # 1st Stage: Prepare strings
        ds_train = ds_train.map(
            lambda batch: {"text": [
                format_zipfit_math_select_prompt_to_prob_soln(
                    problem, solution 
                ) for problem, solution in zip(batch["problem"], batch["solution"])
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
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    return ds_train

