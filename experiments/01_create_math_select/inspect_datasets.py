#!/usr/bin/env python3

from datasets import load_dataset
import pprint
import sys

def inspect_dataset(dataset_name, split=None):
    """Inspect a dataset by loading it and checking its first example."""
    try:
        print(f"\nInspecting {dataset_name}")
        print("=" * 60)
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        print(f"Available splits: {list(dataset.keys())}")
        
        if split is None:
            # If no split is specified, use the first available split
            if len(dataset.keys()) > 0:
                split = list(dataset.keys())[0]
                print(f"Using first available split: {split}")
            else:
                print("No splits available in the dataset.")
                return None
        
        if split not in dataset:
            print(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
            return None
            
        # Get info about the dataset
        print(f"Dataset size: {len(dataset[split])}")
        print(f"Features: {dataset[split].features}")
        
        # Get and print the first example
        example = dataset[split][0]
        print("\nFirst example:")
        pprint.pprint(example)
        
        return dataset
        
    except Exception as e:
        print(f"Error inspecting dataset {dataset_name}: {e}")
        return None

def main():
    # Define source datasets
    src_datasets = [
        # ~1-25% really good data
        ('zipfit/Putnam-AXIOM-for-zip-fit-splits', 'train', 150), 
        ('hoskinson-center/proofnet', 'validation', 185),  # Added the correct proofnet source
        ('brando/olympiad-bench-imo-math-boxed-825-v2-21-08-2024', 'train', 700),

        # ~25-50% somewhat related
        ('brando/hendrycks_math', 'train', 5_500),
        ('TIGER-Lab/MathInstruct', 'train', 3_000), 
        ('TIGER-Lab/WebInstructSub', 'train', 3_000), 
        ('brando/small-open-web-math-dataset-v2', 'train', 9_000),
        ('openai/gsm8k', 'train', 7_473),

        # ~50-75% unrelated or really bad data
        ('brando/small-c4-dataset', 'train', 10_000), 
        ('brando/small-c4-dataset', 'validation', 10_000), 
        ('brando/small-c4-dataset', 'test', 10_000), 
        ('iamtarun/python_code_instructions_18k_alpaca', 'train', 18_612),
        ('brando/random-all-ascii-dataset', 'train', 5_000)
    ]

    # If a specific dataset is specified, only inspect that one
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        split = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Find the matching dataset in the list
        matching_datasets = [ds for ds in src_datasets if ds[0] == dataset_name]
        if matching_datasets:
            for dataset_info in matching_datasets:
                inspect_dataset(dataset_info[0], dataset_info[1])
        else:
            print(f"Inspecting dataset not in list: {dataset_name}")
            inspect_dataset(dataset_name, split)
    else:
        # Inspect all datasets
        for dataset_info in src_datasets:
            inspect_dataset(dataset_info[0], dataset_info[1])

if __name__ == "__main__":
    main() 