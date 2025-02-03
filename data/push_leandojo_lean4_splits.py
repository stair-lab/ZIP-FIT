#!/usr/bin/env python3

# data/push_leandojo_lean4_splits.py

import os
import json
from huggingface_hub import create_repo, upload_file, login, whoami

# ------------------------------
# Log in to Hugging Face Hub.
# ------------------------------
key_file_path = os.path.abspath(os.path.expanduser("~/keys/master_hf_token.txt"))
with open(key_file_path, "r", encoding="utf-8") as f:
    token = f.read().strip()

login(token=token)
os.environ["HUGGINGFACE_TOKEN"] = token
user_info = whoami()
print(f"Currently logged in as: {user_info['name']}\n")

# ------------------------------
# Define file paths and dataset repository.
# ------------------------------
FILES = {
    "train": os.path.expanduser("~/data/leandojo_benchmark_4/random/train.json"),
    "val": os.path.expanduser("~/data/leandojo_benchmark_4/random/val.json"),
    "test": os.path.expanduser("~/data/leandojo_benchmark_4/random/test.json"),
}

HF_REPO = "zipfit/leandojo_benchmark_4_random_splits"  # Single dataset repo for all splits


def push_files_to_hf():
    """
    Pushes train, val, and test splits into the same Hugging Face dataset repository.
    The files will be stored in subdirectories named 'train/', 'val/', and 'test/'.
    """
    create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)

    for split, file_path in FILES.items():
        filename = os.path.basename(file_path)
        remote_path = f"{split}/{filename}"  # Place each split in a subdirectory
        print(f"🚀 Uploading {filename} to {HF_REPO} as {remote_path}...")

        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=remote_path,
            repo_id=HF_REPO,
            repo_type="dataset",
            create_pr=False,
        )
    print(f"✅ Successfully pushed all splits to {HF_REPO}!\n")


def main():
    """Main function to upload all dataset splits into one Hugging Face repository."""
    push_files_to_hf()


if __name__ == "__main__":
    main()
