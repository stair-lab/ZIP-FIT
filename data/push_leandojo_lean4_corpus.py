#!/usr/bin/env python3

# data/push_leandojo_lean4_corpus.py

import os
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
# Define corpus file path and dataset name.
# ------------------------------
FILE_PATH = os.path.expanduser("~/data/leandojo_benchmark_4/corpus.jsonl")
HF_REPO = "zipfit/leandojo_benchmark_4_corpus"  # Upload to the same dataset as splits


def push_corpus_to_hf():
    """
    Pushes the corpus.jsonl file to Hugging Face under the correct repo.
    """
    create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)

    filename = os.path.basename(FILE_PATH)
    print(f"🚀 Uploading {filename} to {HF_REPO}...")

    upload_file(
        path_or_fileobj=FILE_PATH,
        path_in_repo=filename,  # Store corpus.jsonl in the root of the dataset
        repo_id=HF_REPO,
        repo_type="dataset",
        create_pr=False,
    )
    print(f"✅ Successfully pushed {filename} to {HF_REPO}!\n")


def main():
    """Main function to upload the corpus dataset."""
    push_corpus_to_hf()


if __name__ == "__main__":
    main()
