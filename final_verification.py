#!/usr/bin/env python3

from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset

def compare_repo_files(source_repo, target_repo):
    """Compare files between source and target repositories."""
    api = HfApi()
    
    print(f"\nComparing files between {source_repo} and {target_repo}:")
    print("=" * 60)
    
    # Get file lists
    source_files = api.list_repo_files(repo_id=source_repo, repo_type="dataset")
    target_files = api.list_repo_files(repo_id=target_repo, repo_type="dataset")
    
    # Filter out system files
    source_files = [f for f in source_files if not f.startswith('.')]
    target_files = [f for f in target_files if not f.startswith('.')]
    
    # Check if all source files exist in target
    source_missing = [f for f in source_files if f not in target_files]
    if source_missing:
        print(f"⚠️ Files in {source_repo} missing from {target_repo}:")
        for f in source_missing:
            print(f"  - {f}")
    else:
        print(f"✅ All files from {source_repo} exist in {target_repo}")
    
    # Additional files in target
    target_additional = [f for f in target_files if f not in source_files]
    if target_additional:
        print(f"ℹ️ Additional files in {target_repo} (this is expected for generated metadata):")
        for f in target_additional:
            print(f"  - {f}")
    
    return source_files, target_files

def compare_datasets(source_repo, target_repo):
    """Compare dataset structure and content between source and target."""
    print(f"\nComparing dataset content between {source_repo} and {target_repo}:")
    print("=" * 60)
    
    # Load datasets
    source_dataset = load_dataset(source_repo)
    target_dataset = load_dataset(target_repo)
    
    # Compare splits
    source_splits = set(source_dataset.keys())
    target_splits = set(target_dataset.keys())
    
    if source_splits == target_splits:
        print(f"✅ Splits match: {source_splits}")
        
        # Check each split's size
        all_sizes_match = True
        for split in source_splits:
            source_size = len(source_dataset[split])
            target_size = len(target_dataset[split])
            
            if source_size == target_size:
                print(f"✅ Split '{split}' size matches: {source_size} examples")
            else:
                print(f"⚠️ Split '{split}' size mismatch: source={source_size}, target={target_size}")
                all_sizes_match = False
        
        if all_sizes_match:
            print("✅ All split sizes match")
        
        # Check columns/features
        for split in source_splits:
            source_columns = set(source_dataset[split].column_names)
            target_columns = set(target_dataset[split].column_names)
            
            if source_columns == target_columns:
                print(f"✅ Columns for split '{split}' match: {source_columns}")
            else:
                print(f"⚠️ Columns mismatch for split '{split}':")
                print(f"  - Source columns: {source_columns}")
                print(f"  - Target columns: {target_columns}")
                print(f"  - Missing in target: {source_columns - target_columns}")
                print(f"  - Extra in target: {target_columns - source_columns}")
    else:
        print(f"⚠️ Splits mismatch:")
        print(f"  - Source splits: {source_splits}")
        print(f"  - Target splits: {target_splits}")
        print(f"  - Missing in target: {source_splits - target_splits}")
        print(f"  - Extra in target: {target_splits - source_splits}")

def main():
    print("FINAL VERIFICATION OF REPOSITORY TRANSFERS")
    
    repositories = [
        {"source": "UDACA/proofnet-v3-lean4", "target": "brando/proofnet-v3-lean4"},
        {"source": "UDACA/minif2f-lean4", "target": "brando/minif2f-lean4"}
    ]
    
    for repo_info in repositories:
        source_repo = repo_info["source"]
        target_repo = repo_info["target"]
        
        print(f"\n{'#'*70}")
        print(f"# Verifying: {source_repo} -> {target_repo}")
        print(f"{'#'*70}")
        
        # Compare repository files
        compare_repo_files(source_repo, target_repo)
        
        # Compare dataset content
        compare_datasets(source_repo, target_repo)
    
    print("\nVerification completed!")

if __name__ == "__main__":
    main() 