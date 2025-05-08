"""
Script to test loading of math datasets from Hugging Face, particularly
'zipfit/math-select-06062025' and 'zipfit/Putnam-AXIOM-for-zip-fit-splits'.
"""

import sys
from pprint import pprint
from datasets import load_dataset, get_dataset_split_names

def inspect_dataset(dataset_name, split=None):
    """
    Load and inspect a dataset, either a specific split or all available splits.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        split: Specific split to load, or None to load all splits
    """
    print(f"\n{'='*80}")
    print(f"Inspecting dataset: {dataset_name}")
    print(f"{'='*80}")
    
    try:
        # If split is specified, inspect just that split
        if split:
            print(f"\nLoading split: {split}")
            ds = load_dataset(dataset_name, split=split)
            print_dataset_info(ds, split)
        # Otherwise, inspect all available splits
        else:
            # Get all available splits for the dataset
            splits = get_dataset_split_names(dataset_name)
            print(f"Available splits: {splits}")
            
            # Load and inspect each split
            for split in splits:
                print(f"\nLoading split: {split}")
                ds = load_dataset(dataset_name, split=split)
                print_dataset_info(ds, split)
                
        return True
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name} (split={split}): {e}")
        return False

def print_dataset_info(dataset, split_name):
    """
    Print information about a dataset split.
    
    Args:
        dataset: The loaded dataset (or split)
        split_name: Name of the split
    """
    # Print basic information
    print(f"  Split: {split_name}")
    print(f"  Number of examples: {len(dataset)}")
    print(f"  Column names: {dataset.column_names}")
    
    # Print schema information
    print("\n  Sample examples:")
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        print(f"\n  Example {i+1}:")
        for col in dataset.column_names:
            value = example[col]
            # If the value is very long, truncate it
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"    {col}: {value}")
    
    # Print a full example
    if len(dataset) > 0:
        print("\n  First complete example:")
        pprint(dataset[0])

def load_and_verify_putnam_axiom_dataset():
    """
    Load and verify the Putnam-AXIOM dataset, and demonstrate how to use
    it in a training or evaluation pipeline.
    
    Returns:
        dict: A dictionary containing the loaded datasets for each split
    """
    dataset_name = "zipfit/Putnam-AXIOM-for-zip-fit-splits"
    
    try:
        # Get available splits
        splits = get_dataset_split_names(dataset_name)
        print(f"Available splits for {dataset_name}: {splits}")
        
        # Load all splits
        datasets = {}
        for split in splits:
            print(f"Loading {split} split...")
            ds = load_dataset(dataset_name, split=split)
            datasets[split] = ds
            print(f"  Loaded {len(ds)} examples")
            
        # Verify that we can iterate through all examples without issues
        for split_name, ds in datasets.items():
            print(f"Verifying {split_name} split ({len(ds)} examples)...")
            problem_count = 0
            solution_count = 0
            
            # Count examples with problems and solutions
            for i, example in enumerate(ds):
                if example.get('problem'):
                    problem_count += 1
                if example.get('solution'):
                    solution_count += 1
                    
                # Print progress for large datasets
                if (i+1) % 50 == 0 or i+1 == len(ds):
                    print(f"  Processed {i+1}/{len(ds)} examples")
            
            print(f"  Split {split_name}: {problem_count}/{len(ds)} have problems, {solution_count}/{len(ds)} have solutions")
        
        print("\nVerification complete. Dataset loads and can be iterated through successfully.")
        return datasets
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None

def example_usage_for_training():
    """
    Example of how to use the Putnam-AXIOM dataset in a training pipeline.
    """
    print("\n=== Example Usage for Training ===")
    
    # Load the datasets
    dataset_name = "zipfit/Putnam-AXIOM-for-zip-fit-splits"
    try:
        ds_train = load_dataset(dataset_name, split="train")
        ds_val = load_dataset(dataset_name, split="validation")
        print(f"Loaded training set ({len(ds_train)} examples) and validation set ({len(ds_val)} examples)")
        
        # Try to import transformers for tokenization example
        try:
            from transformers import AutoTokenizer
            
            # Example of tokenizing with a model
            model_name = "gpt2"  # This is just an example, use your actual model
            print(f"Example tokenization with {model_name} tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                # Some tokenizers (like GPT-2) don't have a pad token by default
                tokenizer.pad_token = tokenizer.eos_token
            
            # Example of creating a prompt
            example = ds_train[0]
            prompt = f"Problem: {example['problem']}\nSolution:"
            print(f"\nExample prompt:\n{prompt}")
            
            # Tokenize
            tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
            print(f"Tokenized to {len(tokenized['input_ids'])} tokens")
            
        except ImportError:
            print("\nNote: 'transformers' library not found. Skipping tokenization example.")
            print("To install: pip install transformers")
            
            # Show a basic example without tokenization
            example = ds_train[0]
            prompt = f"Problem: {example['problem']}\nSolution:"
            print(f"\nExample prompt that could be used for training:\n{prompt}")
        
        # Example of how you might prepare this for training
        print("\nTo use this dataset for fine-tuning, you would typically:")
        print("1. Format each example with an appropriate prompt template")
        print("2. Tokenize the inputs and targets")
        print("3. Use the tokenized dataset with a Trainer or custom training loop")
        
        return True
        
    except Exception as e:
        print(f"Error in example usage: {e}")
        return False

def example_minimal_usage():
    """
    Minimal example of how to use the Putnam-AXIOM dataset without additional dependencies.
    """
    print("\n=== Minimal Example Usage ===")
    
    # Load the datasets
    dataset_name = "zipfit/Putnam-AXIOM-for-zip-fit-splits"
    try:
        # Load all splits
        ds_train = load_dataset(dataset_name, split="train")
        ds_val = load_dataset(dataset_name, split="validation")
        ds_test = load_dataset(dataset_name, split="test")
        
        print(f"Dataset loaded successfully:")
        print(f"  - Train: {len(ds_train)} examples")
        print(f"  - Validation: {len(ds_val)} examples")
        print(f"  - Test: {len(ds_test)} examples")
        
        # Show how to access a random example
        import random
        random_idx = random.randint(0, len(ds_train) - 1)
        example = ds_train[random_idx]
        
        print(f"\nRandom example from training set (index {random_idx}):")
        print(f"  Problem ID: {example['id']}")
        print(f"  Source: {example['source']}")
        print(f"  Type: {example['type']}")
        
        # Show how to create a simple prompt-completion pair
        prompt = f"Problem: {example['problem']}\n\nSolve this math problem step by step."
        completion = example['solution']
        
        print(f"\nExample prompt-completion pair:")
        print(f"PROMPT:\n{prompt[:200]}...")
        print(f"\nCOMPLETION:\n{completion[:200]}...")
        
        # Show how to filter or map the dataset
        filtered_ds = ds_train.filter(lambda example: "Algebra" in example["type"])
        print(f"\nFiltered dataset (only Algebra problems): {len(filtered_ds)} examples")
        
        # Add a custom field through mapping
        def add_prompt_field(example):
            example["prompt"] = f"Problem: {example['problem']}\nSolution:"
            return example
            
        mapped_ds = ds_train.map(add_prompt_field)
        print(f"Added 'prompt' field to dataset. New columns: {mapped_ds.column_names}")
        
        return True
        
    except Exception as e:
        print(f"Error in minimal example: {e}")
        return False

def main():
    # Test loading the Putnam-AXIOM dataset
    print("\nTesting loading of zipfit/Putnam-AXIOM-for-zip-fit-splits dataset...")
    datasets = load_and_verify_putnam_axiom_dataset()
    
    if datasets:
        # Run the minimal example first (doesn't require transformers)
        example_minimal_usage()
        
        # Then try the training example (which has a fallback if transformers is missing)
        example_usage_for_training()
    
    # You can also use the original inspect_dataset function if needed
    # inspect_dataset("zipfit/math-select-06062025")
    # inspect_dataset("zipfit/Putnam-AXIOM-for-zip-fit-splits")

if __name__ == "__main__":
    main()
