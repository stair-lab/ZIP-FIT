#!/usr/bin/env python3

import os
import json
from huggingface_hub import login, HfApi, hf_hub_download
from datasets import load_dataset
from collections import defaultdict

def main():
    # Config
    repo_id = "zipfit/math-select-06062025"
    token_path = "/home/brandomiranda/keys/master_hf_token.txt"
    
    # Read token
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    # Login to HF
    login(token=token)
    api = HfApi(token=token)
    
    # Try to download current README, if it exists
    try:
        readme_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="dataset",
            token=token
        )
        with open(readme_path, 'r') as f:
            current_readme = f.read()
        print("Downloaded existing README")
    except:
        current_readme = ""
        print("No existing README found or error downloading")
    
    # Load the dataset to get source information
    print("Loading dataset to get source information...")
    dataset = load_dataset(repo_id)
    
    # Group examples by source
    sources = defaultdict(int)
    for item in dataset['src']:
        sources[item['original_source']] += 1
    
    # Create new README
    print("Creating updated README...")
    readme_path = "/tmp/updated_readme.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# Math Selection Dataset

This dataset contains {len(dataset['src'])} examples from various math and general sources for data selection experiments. Each example contains:
- `text`: The content, with math examples following the "Problem:\\n...\\n\\nSolution:\\n..." format
- `original_source`: Source information for the example

## Formats Available
- **Parquet**: Available in the `data/` directory
- **Jsonlines**: Available as `src.jsonl` in the root directory

## Sources:
""")
        # Add source counts
        for source, count in sorted(sources.items()):
            f.write(f"- {source}: {count} examples\n")
    
    # Upload the updated README
    print("\nUploading updated README...")
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"✅ Successfully updated README in {repo_id}")
    except Exception as e:
        print(f"❌ Error updating README: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 