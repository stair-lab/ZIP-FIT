#!/usr/bin/env python3

import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, CommitOperationAdd

# -----------------------------------------------------------------------------
# Step 1: Process original JSONL to create proofnet_lean4_v2.jsonl
# -----------------------------------------------------------------------------
INPUT_FILE = "/lfs/skampere1/0/brando9/ZIP-FIT/data/old/proofnet.jsonl"
OUTPUT_FILE = "/lfs/skampere1/0/brando9/ZIP-FIT/data/proofnet_lean4_v2.jsonl"

def clean_informal_prefix(text: str) -> str:
    """Removes '/--' and '-/' from the given text."""
    return text.replace('/--', '').replace('-/', '').strip()

def process_jsonl(input_path: str, output_path: str):
    """Creates a new JSONL with 'nl_statement' field from 'informal_prefix'."""
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            if "informal_prefix" in data:
                data["nl_statement"] = clean_informal_prefix(data["informal_prefix"])
            outfile.write(json.dumps(data) + "\n")

# -----------------------------------------------------------------------------
# Step 2: Load the new JSONL into a DatasetDict (validation/test)
# -----------------------------------------------------------------------------
def create_dataset_splits(file_path: str) -> DatasetDict:
    """Loads JSONL into a huggingface Dataset, then filters by 'split'."""
    with open(file_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    full_ds = Dataset.from_list(items)
    ds_validation = full_ds.filter(lambda x: x.get("split") == "valid")
    ds_test       = full_ds.filter(lambda x: x.get("split") == "test")

    return DatasetDict({
        "validation": ds_validation,
        "test": ds_test
    })

# -----------------------------------------------------------------------------
# Step 3 & 4: Push dataset to HF Hub + (optional) create a minimal README
# -----------------------------------------------------------------------------
def main():
    # 1) Create the proofnet_lean4_v2.jsonl file
    process_jsonl(INPUT_FILE, OUTPUT_FILE)
    print(f"Processed dataset saved to {OUTPUT_FILE}")

    # 2) Load dataset splits
    ds_splits = create_dataset_splits(OUTPUT_FILE)
    print("Number of examples per split:", ds_splits.num_rows)

    # 3) Push to HF Hub
    repo_id = "UDACA/proofnet-v2-lean4"  # Change if needed
    ds_splits.push_to_hub(
        repo_id=repo_id,
        private=False,  # set True if you want a private repo
        token=None      # or "hf_yourToken" if you prefer explicitly
    )
    print(f"Pushed dataset to: https://huggingface.co/datasets/{repo_id}")

    # 4) (Optional) Create & commit a minimal README.md
    # Remove or comment out this block if you don't want a README.
    readme_text = """# ProofNet Lean4 v2

A Lean 4 version of the ProofNet dataset.  
We provide two splits: `validation` and `test`.  

Adds a `nl_statement` field which is a cleaned version of the original `informal_prefix`.
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
