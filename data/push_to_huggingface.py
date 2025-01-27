"""
push_to_huggingface.py

A script to upload local JSONL files to a Hugging Face Dataset repository
under the user 'UDACA'. Adjust parameters, dataset names, and file paths
as necessary.

Usage:
  python push_to_huggingface.py
"""

import os
from huggingface_hub import create_repo, upload_file, whoami

def push_files_to_hf(
    repo_id: str,
    file_list: list[str],
    repo_type: str = "dataset",
    create_pr: bool = False
) -> None:
    """
    Pushes a list of local files to a Hugging Face repository.

    :param repo_id: The name of the repository on Hugging Face, e.g., "UDACA/minif2f".
    :param file_list: List of file paths (strings) to be uploaded.
    :param repo_type: The type of repository (defaults to 'dataset').
    :param create_pr: If True, create a Pull Request instead of pushing directly.
                     Requires push access.
    """
    # Create the repository if it doesn't already exist.
    # 'exist_ok=True' means it won't fail if the repo is already there.
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    # For each file in file_list, upload to the corresponding path in the repo
    # (by default we place it at the root with the same filename).
    for local_file in file_list:
        filename = os.path.basename(local_file)
        print(f"Uploading {filename} to {repo_id}...")
        upload_file(
            path_or_fileobj=local_file,   # local path
            path_in_repo=filename,        # destination path in the repo
            repo_id=repo_id,
            repo_type=repo_type,
            create_pr=create_pr
        )
    print(f"Successfully pushed files to {repo_id}!\n")


def main() -> None:
    """
    Main driver function to demonstrate pushing multiple JSONL files to
    different Hugging Face dataset repos for 'UDACA'.
    """
    # Check your HF login information
    me = whoami()
    print(f"Currently logged in as: {me['name']}\n")

    # Files for minif2f dataset
    minif2f_files = [
        "minif2f.jsonl",              # Adjust if in a subfolder, e.g. "data/minif2f.jsonl"
        "minif2f_valid_few_shot.jsonl"
    ]
    # Files for proofnet dataset
    proofnet_files = [
        "proofnet.jsonl"
    ]

    # Adjust these repos to your needs. The user in this example is "UDACA".
    # If you want each set in a separate dataset, keep them separate.
    # If you want them in the same dataset, use the same `repo_id`.
    minif2f_repo_id = "UDACA/minif2f"   # or "UDACA/minif2f-v2" if you prefer
    proofnet_repo_id = "UDACA/proofnet"

    # Push minif2f files
    push_files_to_hf(
        repo_id=minif2f_repo_id,
        file_list=minif2f_files,
        repo_type="dataset",
        create_pr=False  # Set this to True if you prefer creating a PR.
    )

    # Push proofnet file
    push_files_to_hf(
        repo_id=proofnet_repo_id,
        file_list=proofnet_files,
        repo_type="dataset",
        create_pr=False
    )

if __name__ == "__main__":
    main()
