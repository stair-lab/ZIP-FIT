#!/usr/bin/env python3
"""
split_and_push.py

A script to:
1) Log into Hugging Face using a local token file.
2) Separate JSONL data by 'split' field into multiple files (train, valid, test, etc.).
3) Push those files to Hugging Face datasets under your user/repo, adding "lean4" to dataset names.

Usage:
  python split_and_push.py
"""

import json
import os
from collections import defaultdict
from typing import List, Dict

# Hugging Face Hub libraries
from huggingface_hub import create_repo, upload_file, login, whoami


def separate_by_split(
    input_files: List[str],
    output_dir: str
) -> Dict[str, List[str]]:
    """
    Reads each JSONL file in `input_files`, and separates lines by the 'split'
    field, writing them into split-specific files inside `output_dir`.

    :param input_files: A list of JSONL file paths to be read.
    :param output_dir: Directory where the split files will be saved.
    :return: A dictionary mapping each dataset name (e.g. 'minif2f') to a list
             of output file paths that were created for that dataset.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # data_by_dataset maps: dataset_name -> {split_name -> list of JSON records}
    data_by_dataset: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))

    for file_path in input_files:
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Could not find file: {file_path}")

        # Infer a dataset name from the file name (customize logic if needed)
        file_name = os.path.basename(file_path)
        dataset_name = file_name.replace(".jsonl", "")
        # Example: "minif2f.jsonl" -> "minif2f"

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # The record must contain a "split" key. If missing, treat as "unspecified".
                split_name = record.get("split", "unspecified")
                data_by_dataset[dataset_name][split_name].append(record)

    # Now we have data grouped by dataset and split.
    # Write out each dataset's splits into separate files, e.g.:
    #   output_dir / minif2f / train.jsonl
    #   output_dir / minif2f / valid.jsonl
    output_files_map: Dict[str, List[str]] = defaultdict(list)

    for ds_name, split_dict in data_by_dataset.items():
        ds_dir = os.path.join(output_dir, ds_name)
        os.makedirs(ds_dir, exist_ok=True)

        # For each split in this dataset, write a new JSONL
        for split_name, rows in split_dict.items():
            out_file = os.path.join(ds_dir, f"{split_name}.jsonl")
            with open(out_file, "w", encoding="utf-8") as outf:
                for row in rows:
                    outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_files_map[ds_name].append(out_file)

    return output_files_map


def push_files_to_hf(
    repo_id: str,
    file_list: List[str],
    path_prefix: str = "",
    repo_type: str = "dataset",
    create_pr: bool = False
) -> None:
    """
    Pushes a list of local files to a Hugging Face repository (dataset by default).

    :param repo_id: The name of the repository on Hugging Face, e.g., "UDACA/minif2f".
    :param file_list: List of file paths (strings) to be uploaded.
    :param path_prefix: An optional subdirectory path in the repo to place the files.
    :param repo_type: The type of repository (defaults to 'dataset').
    :param create_pr: If True, create a Pull Request instead of pushing directly.
    """
    # Create (or access) the repository.
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    # Upload each file
    for local_file in file_list:
        local_file = os.path.abspath(os.path.expanduser(local_file))
        if not os.path.isfile(local_file):
            raise FileNotFoundError(f"File does not exist: {local_file}")

        filename = os.path.basename(local_file)
        remote_path = os.path.join(path_prefix, filename) if path_prefix else filename
        print(f"Uploading {local_file} to {repo_id} as {remote_path} ...")

        upload_file(
            path_or_fileobj=local_file,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            create_pr=create_pr
        )
    print(f"Successfully pushed files to '{repo_id}'!\n")


def main() -> None:
    """
    Main driver function:
      1) Logs into Hugging Face with a local token.
      2) Separates JSONL lines by 'split' field.
      3) Pushes them to user-specified HF dataset repos and prints URLs.
    """
    # ----------
    # 0) Log In
    # ----------
    key_file_path = "~/keys/master_hf_token.txt"
    key_file_path = os.path.abspath(os.path.expanduser(key_file_path))
    if not os.path.isfile(key_file_path):
        raise FileNotFoundError(f"Could not find token file: {key_file_path}")

    with open(key_file_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    login(token=token)

    user_info = whoami()
    print(f"Currently logged in as: {user_info['name']}\n")

    # ----------
    # 1) Separate JSONL files into train/valid/test splits
    #    Adjust these absolute paths to match your local environment
    # ----------
    input_files = [
        "~/ZIP-FIT/data/minif2f.jsonl",
        "~/ZIP-FIT/data/minif2f_valid_few_shot.jsonl",
        "~/ZIP-FIT/data/proofnet.jsonl"
    ]

    # Use absolute path for the output directory as well
    output_directory = os.path.abspath(os.path.expanduser("~/ZIP-FIT/data/split_data"))
    os.makedirs(output_directory, exist_ok=True)

    dataset_files_map = separate_by_split(input_files, output_directory)

    # ----------
    # 2) Define where to push them (dataset name -> repo ID with "lean4" suffix)
    # ----------
    dataset_to_repo = {
        "minif2f": "UDACA/minif2f-lean4",
        "minif2f_valid_few_shot": "UDACA/minif2f-lean4",
        "proofnet": "UDACA/proofnet-lean4"
    }

    # ----------
    # 3) Push each dataset's splitted files
    #    Print the URL after each push so you can check
    # ----------
    for ds_name, files_list in dataset_files_map.items():
        if ds_name not in dataset_to_repo:
            print(f"[Warning] No repo mapping for dataset '{ds_name}'. Skipping push.")
            continue

        repo_id = dataset_to_repo[ds_name]
        # Optionally store them under a subfolder in the repo (named after ds_name)
        path_prefix = ds_name

        print(f"\n=== Pushing dataset: '{ds_name}' => '{repo_id}' ===")
        push_files_to_hf(
            repo_id=repo_id,
            file_list=files_list,
            path_prefix=path_prefix,
            repo_type="dataset",
            create_pr=False
        )

        # Print the URL so you can easily check after each push
        print(f"Check it out at: https://huggingface.co/datasets/{repo_id}\n")


if __name__ == "__main__":
    main()
