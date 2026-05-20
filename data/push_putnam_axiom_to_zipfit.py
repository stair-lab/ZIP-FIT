#!/usr/bin/env python3

import json
import os
import random
from pathlib import Path
import datasets
from datasets import Dataset
from huggingface_hub import login

def load_json_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    """Add year and variation fields to the data."""
    for item in data:
        # Extract year from ID
        if '_' in item['id']:
            year = item['id'].split('_')[0]
            item['year'] = year
        else:
            item['year'] = ""
        
        # Add variation field (these are not variations)
        item['variation'] = 0
        
        # Standardize the keys
        # If 'original solution' exists but 'original_solution' doesn't, copy it
        if 'original solution' in item and 'original_solution' not in item:
            item['original_solution'] = item['original solution']
            del item['original solution']
        # If neither exists, add an empty one
        elif 'original_solution' not in item:
            item['original_solution'] = ""
    
    return data

def create_splits(data, val_size=150, seed=42):
    """Create dev and test splits that together cover the full dataset.
    
    Args:
        data: The entire dataset
        val_size: The size of the validation set
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with full_eval, test, and validation splits
    """
    random.seed(seed)
    random.shuffle(data)
    
    # Use fixed sizes as requested
    val_data = data[:val_size]
    test_data = data[val_size:]
    
    return {
        "full_eval": data,    # All data
        "test": test_data,    # 372 examples
        "validation": val_data  # 150 examples - renamed from "val" to "validation"
    }

def dataset_to_hf_format(data_list):
    """Convert the list of dictionaries to the format expected by Hugging Face datasets."""
    # Extract all keys from the first item
    if not data_list:
        return Dataset.from_dict({})
    
    # Initialize dataset with empty lists for each key
    dataset_dict = {}
    for item in data_list:
        for key in item:
            if key not in dataset_dict:
                dataset_dict[key] = []
    
    # Fill the dataset dictionary
    for item in data_list:
        for key in dataset_dict:
            if key in item:
                dataset_dict[key].append(item[key])
            else:
                # Handle missing keys with an empty string
                dataset_dict[key].append("")
    
    return Dataset.from_dict(dataset_dict)

def push_to_huggingface(splits, repo_id, token_path):
    """Push the splits to Hugging Face."""
    # Read the token
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    # Login to Hugging Face
    login(token=token)
    
    # Create a DatasetDict with the splits
    dataset_dict = {}
    for split_name, split_data in splits.items():
        dataset_dict[split_name] = dataset_to_hf_format(split_data)
    
    # Create a datasets.DatasetDict from the dictionary
    hf_dataset = datasets.DatasetDict(dataset_dict)
    
    # Push to Hugging Face
    hf_dataset.push_to_hub(repo_id)
    
    print(f"Successfully pushed dataset with {len(splits)} splits to {repo_id}")
    for split_name, split_data in splits.items():
        print(f"- {split_name}: {len(split_data)} examples")

def main():
    # Paths
    data_path = "/home/brandomiranda/ZIP-FIT/data/Putnam_AXIOM_Original_v3.json"
    token_path = "/home/brandomiranda/keys/master_hf_token.txt"
    repo_id = "zipfit/Putnam-AXIOM-for-zip-fit-splits"
    
    # Load the data
    print(f"Loading data from {data_path}...")
    data = load_json_data(data_path)
    print(f"Loaded {len(data)} examples")
    
    # Preprocess data
    print("Preprocessing data...")
    data = preprocess_data(data)
    
    # Create splits
    print("Creating splits...")
    splits = create_splits(data)
    
    # Push to Hugging Face
    print(f"Pushing to Hugging Face repository: {repo_id}")
    push_to_huggingface(splits, repo_id, token_path)
    
    print("Done!")

if __name__ == "__main__":
    main() 