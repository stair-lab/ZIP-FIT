#!/usr/bin/env python3

import os
import json
from huggingface_hub import login, HfApi, hf_hub_download

def main():
    # Config
    orig_repo = "zipfit/math-select-06062025"
    jsonl_repo = "zipfit/math-select-06062025-jsonl"
    jsonl_file = "/tmp/math_select_src.jsonl"
    token_path = "/home/brandomiranda/keys/master_hf_token.txt"
    
    # Read token
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    # Login to HF
    login(token=token)
    api = HfApi(token=token)
    
    # First, check if we need to regenerate the jsonl file
    if not os.path.exists(jsonl_file):
        print(f"Jsonl file not found at {jsonl_file}, recreating it...")
        dataset = load_dataset(orig_repo)
        
        # Export to jsonlines
        print(f"Exporting dataset to {jsonl_file}...")
        with open(jsonl_file, 'w') as f:
            for item in dataset['src']:
                f.write(json.dumps(item) + '\n')
        
        print(f"Exported jsonl file to {jsonl_file}")
    else:
        print(f"Using existing jsonl file at {jsonl_file}")
    
    # Upload jsonl file to the original repo
    print(f"\nUploading jsonl file to the original repo ({orig_repo})...")
    try:
        api.upload_file(
            path_or_fileobj=jsonl_file,
            path_in_repo="src.jsonl",
            repo_id=orig_repo,
            repo_type="dataset"
        )
        print(f"✅ Successfully uploaded jsonl file to {orig_repo}")
    except Exception as e:
        print(f"❌ Error uploading to original repo: {e}")
    
    # Delete the separate jsonl repo
    print(f"\nDeleting the separate jsonl repo ({jsonl_repo})...")
    try:
        api.delete_repo(
            repo_id=jsonl_repo,
            repo_type="dataset"
        )
        print(f"✅ Successfully deleted {jsonl_repo}")
    except Exception as e:
        print(f"❌ Error deleting jsonl repo: {e}")
    
    print("\nDone! Everything is now consolidated in the original repo.")

if __name__ == "__main__":
    from datasets import load_dataset
    main() 