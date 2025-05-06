#!/usr/bin/env python3

import os
from datasets import load_dataset
from huggingface_hub import login, HfApi

def main():
    # Paths and configuration
    token_path = "/home/brandomiranda/keys/master_hf_token.txt"
    source_datasets = [
        {"source": "UDACA/proofnet-v3-lean4", "target": "brando/proofnet-v3-lean4"},
        {"source": "UDACA/minif2f-lean4", "target": "brando/minif2f-lean4"}
    ]
    
    # Read the token
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    # Login to Hugging Face
    login(token=token)
    api = HfApi(token=token)
    
    for dataset_info in source_datasets:
        source_repo = dataset_info["source"]
        target_repo = dataset_info["target"]
        
        print(f"Processing: {source_repo} -> {target_repo}")
        
        # Load source dataset
        print(f"Loading dataset from {source_repo}...")
        dataset = load_dataset(source_repo)
        print(f"Loaded dataset with splits: {list(dataset.keys())}")
        
        # Create the target repo if it doesn't exist
        try:
            api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)
            print(f"Created/verified target repo: {target_repo}")
        except Exception as e:
            print(f"Error creating repo {target_repo}: {e}")
            continue
        
        # Push to target
        print(f"Pushing dataset to {target_repo}...")
        try:
            dataset.push_to_hub(target_repo)
            print(f"Successfully pushed {source_repo} to {target_repo}")
        except Exception as e:
            print(f"Error pushing to {target_repo}: {e}")
            continue
        
        # Verify the push
        try:
            # Try to load the dataset to verify
            verification = load_dataset(target_repo)
            
            # Check if splits are the same
            source_splits = set(dataset.keys())
            target_splits = set(verification.keys())
            
            if source_splits == target_splits:
                print(f"✅ Verified: {target_repo} has the same splits as {source_repo}")
                
                # Check sizes of each split
                sizes_match = True
                for split in source_splits:
                    source_size = len(dataset[split])
                    target_size = len(verification[split])
                    if source_size != target_size:
                        print(f"⚠️ Size mismatch for split '{split}': source={source_size}, target={target_size}")
                        sizes_match = False
                
                if sizes_match:
                    print(f"✅ All split sizes match between {source_repo} and {target_repo}")
                
            else:
                print(f"⚠️ Splits mismatch: {source_repo}={source_splits}, {target_repo}={target_splits}")
                
        except Exception as e:
            print(f"Error verifying dataset {target_repo}: {e}")
    
    print("Script completed.")

if __name__ == "__main__":
    main() 