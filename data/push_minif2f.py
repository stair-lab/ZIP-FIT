#!/usr/bin/env python3

import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, CommitOperationAdd


# -----------------------------------------------------------------------------
# Step 1: Define paths to your two JSONL files: valid and test
# -----------------------------------------------------------------------------
VALID_FILE = "/lfs/skampere1/0/brando9/ZIP-FIT/data/minif2f_valid.jsonl"
TEST_FILE  = "/lfs/skampere1/0/brando9/ZIP-FIT/data/minif2f_test.jsonl"


# -----------------------------------------------------------------------------
# Utility: Convert all row values to strings (prevents PyArrow float errors)
# -----------------------------------------------------------------------------
def convert_all_to_strings(list_of_dicts):
    """Cast all values in each row to str."""
    for record in list_of_dicts:
        for key, value in record.items():
            record[key] = str(value)
    return list_of_dicts


# -----------------------------------------------------------------------------
# Step 2: Create a DatasetDict from the validation and test JSONL files
# -----------------------------------------------------------------------------
def create_dataset_splits(valid_path: str, test_path: str) -> DatasetDict:
    """
    Loads two JSONL files and makes them into a DatasetDict
    with 'validation' and 'test' splits.
    """
    # Load validation data
    with open(valid_path, "r", encoding="utf-8") as f_valid:
        valid_data = [json.loads(line) for line in f_valid]
    # Cast everything to string
    valid_data = convert_all_to_strings(valid_data)
    ds_validation = Dataset.from_list(valid_data)

    # Load test data
    with open(test_path, "r", encoding="utf-8") as f_test:
        test_data = [json.loads(line) for line in f_test]
    # Cast everything to string
    test_data = convert_all_to_strings(test_data)
    ds_test = Dataset.from_list(test_data)

    # Return a DatasetDict
    return DatasetDict({
        "validation": ds_validation,
        "test": ds_test
    })


# -----------------------------------------------------------------------------
# Step 3 & 4: Push dataset to HF Hub + optionally create a minimal README
# -----------------------------------------------------------------------------
def main():
    # 1) No data processing needed, just use the valid/test files directly.
    print("No data processing step. Using existing JSONL files.")

    # 2) Load dataset splits
    ds_splits = create_dataset_splits(VALID_FILE, TEST_FILE)
    print("Number of examples per split:", ds_splits.num_rows)

    # 3) Push to HF Hub
    repo_id = "UDACA/minif2f-lean4"  # Change if needed
    ds_splits.push_to_hub(
        repo_id=repo_id,
        private=False,   # Set True if you want a private dataset
        token=None       # Or "hf_YourAccessTokenHere"
    )
    print(f"Pushed dataset to: https://huggingface.co/datasets/{repo_id}")

    # 4) (Optional) Create & commit a minimal README.md
    readme_text = """# MiniF2F Lean4

This dataset provides two splits (validation and test) of the MiniF2F dataset adapted for Lean4.

It includes fields such as:
- `id`
- `split`
- `formal_statement`
- `header`
- `nl_statement`
- `nl_proof`
"""
    api = HfApi()
    operations = [
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=readme_text.encode("utf-8")
        )
    ]
    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="Add minimal README",
        repo_type="dataset"
    )
    print(f"README added at: https://huggingface.co/datasets/{repo_id}/blob/main/README.md")


if __name__ == "__main__":
    main()
