#!/usr/bin/env python3

import os
import json
from pathlib import Path
import tempfile
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import login

# Define the math training prompt template function
def get_zipfit_math_train_prompt(problem, solution):
    """Format problem and solution according to the template."""
    template = "Problem:\n{$PROBLEM}\n\nSolution:\n{$SOLUTION}\n\n"
    return template.replace("{$PROBLEM}", problem).replace("{$SOLUTION}", solution)

def process_putnam_axiom(dataset, dataset_info):
    """Process Putnam-AXIOM dataset."""
    print(f"Processing {dataset_info}...")
    dataset_name, split, count = dataset_info
    
    processed_data = []
    
    # Limit to specified count
    dataset = dataset[split].select(range(min(count, len(dataset[split]))))
    
    for item in tqdm(dataset):
        # Get problem and solution
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        
        # Format text using the template
        text = get_zipfit_math_train_prompt(problem, solution)
        
        # Create entry with text and original_source
        entry = {
            "text": text,
            "original_source": f"{dataset_name},{split},{count}"
        }
        
        processed_data.append(entry)
    
    return processed_data

def process_proofnet(dataset, dataset_info):
    """Process proofnet dataset."""
    print(f"Processing {dataset_info}...")
    dataset_name, split, count = dataset_info
    
    processed_data = []
    
    # Limit to specified count
    dataset = dataset[split].select(range(min(count, len(dataset[split]))))
    
    for item in tqdm(dataset):
        # Get nl_statement as problem and nl_proof as solution
        problem = item.get('nl_statement', '')
        solution = item.get('nl_proof', '')
        
        # Verify we have both problem and solution
        if not problem or not solution:
            print(f"Warning: Missing problem or solution for item {item.get('id', 'unknown')}")
            continue
        
        # Format text using the template
        text = get_zipfit_math_train_prompt(problem, solution)
        
        # Create entry with text and original_source
        entry = {
            "text": text,
            "original_source": f"{dataset_name},{split},{count}"
        }
        
        processed_data.append(entry)
    
    return processed_data

def process_olympiad(dataset, dataset_info):
    """Process olympiad dataset."""
    print(f"Processing {dataset_info}...")
    dataset_name, split, count = dataset_info
    
    processed_data = []
    
    # Limit to specified count
    dataset = dataset[split].select(range(min(count, len(dataset[split]))))
    
    for item in tqdm(dataset):
        # Get problem and solution
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        
        # Format text using the template
        text = get_zipfit_math_train_prompt(problem, solution)
        
        # Create entry with text and original_source
        entry = {
            "text": text,
            "original_source": f"{dataset_name},{split},{count}"
        }
        
        processed_data.append(entry)
    
    return processed_data

def process_hendrycks_math(dataset, dataset_info):
    """Process hendrycks_math dataset."""
    print(f"Processing {dataset_info}...")
    dataset_name, split, count = dataset_info
    
    processed_data = []
    
    # Limit to specified count
    dataset = dataset[split].select(range(min(count, len(dataset[split]))))
    
    for item in tqdm(dataset):
        # Get problem and solution
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        
        # Format text using the template
        text = get_zipfit_math_train_prompt(problem, solution)
        
        # Create entry with text and original_source
        entry = {
            "text": text,
            "original_source": f"{dataset_name},{split},{count}"
        }
        
        processed_data.append(entry)
    
    return processed_data

def process_math_instruct(dataset, dataset_info):
    """Process MathInstruct dataset."""
    print(f"Processing {dataset_info}...")
    dataset_name, split, count = dataset_info
    
    processed_data = []
    
    # Limit to specified count
    dataset = dataset[split].select(range(min(count, len(dataset[split]))))
    
    for item in tqdm(dataset):
        # Use instruction as problem and output as solution
        problem = item.get('instruction', '')
        solution = item.get('output', '')
        
        # Format text using the template
        text = get_zipfit_math_train_prompt(problem, solution)
        
        # Create entry with text and original_source
        entry = {
            "text": text,
            "original_source": f"{dataset_name},{split},{count}"
        }
        
        processed_data.append(entry)
    
    return processed_data

def process_web_instruct(dataset, dataset_info):
    """Process WebInstructSub dataset."""
    print(f"Processing {dataset_info}...")
    dataset_name, split, count = dataset_info
    
    processed_data = []
    
    # Limit to specified count
    dataset = dataset[split].select(range(min(count, len(dataset[split]))))
    
    for item in tqdm(dataset):
        # Use question as problem and answer as solution
        problem = item.get('question', '')
        solution = item.get('answer', '')
        
        # Format text using the template
        text = get_zipfit_math_train_prompt(problem, solution)
        
        # Create entry with text and original_source
        entry = {
            "text": text,
            "original_source": f"{dataset_name},{split},{count}"
        }
        
        processed_data.append(entry)
    
    return processed_data

def process_text_only_dataset(dataset, dataset_info):
    """Process datasets that already have a text field."""
    print(f"Processing {dataset_info}...")
    dataset_name, split, count = dataset_info
    
    processed_data = []
    
    # Limit to specified count
    dataset = dataset[split].select(range(min(count, len(dataset[split]))))
    
    for item in tqdm(dataset):
        # Dataset already has a text field
        text = item.get('text', '')
        
        # Create entry with text and original_source
        entry = {
            "text": text,
            "original_source": f"{dataset_name},{split},{count}"
        }
        
        processed_data.append(entry)
    
    return processed_data

def process_gsm8k(dataset, dataset_info):
    """Process GSM8K dataset."""
    print(f"Processing {dataset_info}...")
    dataset_name, split, count = dataset_info
    
    processed_data = []
    
    # Limit to specified count
    dataset = dataset[split].select(range(min(count, len(dataset[split]))))
    
    for item in tqdm(dataset):
        # Get question as problem and answer as solution
        problem = item.get('question', '')
        solution = item.get('answer', '')
        
        # Format text using the template
        text = get_zipfit_math_train_prompt(problem, solution)
        
        # Create entry with text and original_source
        entry = {
            "text": text,
            "original_source": f"{dataset_name},{split},{count}"
        }
        
        processed_data.append(entry)
    
    return processed_data

def process_python_code_instructions(dataset, dataset_info):
    """Process python_code_instructions dataset."""
    print(f"Processing {dataset_info}...")
    dataset_name, split, count = dataset_info
    
    processed_data = []
    
    # Limit to specified count
    dataset = dataset[split].select(range(min(count, len(dataset[split]))))
    
    for item in tqdm(dataset):
        # Use prompt field directly as text
        text = item.get('prompt', '')
        
        # If prompt is missing or empty, construct it from instruction and output
        if not text:
            instruction = item.get('instruction', '')
            output = item.get('output', '')
            text = get_zipfit_math_train_prompt(instruction, output)
        
        # Create entry with text and original_source
        entry = {
            "text": text,
            "original_source": f"{dataset_name},{split},{count}"
        }
        
        processed_data.append(entry)
    
    return processed_data

def main():
    # Define source datasets
    src_datasets = [
        # ~1-25% really good data
        ('zipfit/Putnam-AXIOM-for-zip-fit-splits', 'train', 150), 
        ('hoskinson-center/proofnet', 'validation', 185),  # Fixed: use hoskinson-center/proofnet instead of brando's
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
    
    # Path for token
    token_path = "/home/brandomiranda/keys/master_hf_token.txt"
    output_repo = "zipfit/math-select-06062025"
    
    # Read token
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    # Login to HF
    login(token=token)
    
    # Process all datasets
    all_processed_data = []
    
    for dataset_info in src_datasets:
        dataset_name, split, count = dataset_info
        try:
            # Load dataset
            if dataset_name == 'openai/gsm8k':
                # GSM8K requires specifying a config
                dataset = load_dataset(dataset_name, 'main')
            else:
                dataset = load_dataset(dataset_name)
            
            # Process based on dataset type
            if dataset_name == 'zipfit/Putnam-AXIOM-for-zip-fit-splits':
                processed = process_putnam_axiom(dataset, dataset_info)
            elif dataset_name == 'hoskinson-center/proofnet':
                processed = process_proofnet(dataset, dataset_info)
            elif dataset_name == 'brando/olympiad-bench-imo-math-boxed-825-v2-21-08-2024':
                processed = process_olympiad(dataset, dataset_info)
            elif dataset_name == 'brando/hendrycks_math':
                processed = process_hendrycks_math(dataset, dataset_info)
            elif dataset_name == 'TIGER-Lab/MathInstruct':
                processed = process_math_instruct(dataset, dataset_info)
            elif dataset_name == 'TIGER-Lab/WebInstructSub':
                processed = process_web_instruct(dataset, dataset_info)
            elif dataset_name == 'brando/small-open-web-math-dataset-v2' or dataset_name == 'brando/small-c4-dataset' or dataset_name == 'brando/random-all-ascii-dataset':
                processed = process_text_only_dataset(dataset, dataset_info)
            elif dataset_name == 'openai/gsm8k':
                processed = process_gsm8k(dataset, dataset_info)
            elif dataset_name == 'iamtarun/python_code_instructions_18k_alpaca':
                processed = process_python_code_instructions(dataset, dataset_info)
            else:
                print(f"Unknown dataset type: {dataset_name}. Skipping...")
                continue
                
            all_processed_data.extend(processed)
            print(f"Processed {len(processed)} examples from {dataset_info}")
            
        except Exception as e:
            print(f"Error processing {dataset_info}: {e}")
            continue
    
    print(f"Total processed examples: {len(all_processed_data)}")
    
    # Create HF dataset
    hf_dataset = Dataset.from_list(all_processed_data)
    
    # Save a sample to inspect
    sample_path = "/tmp/math_dataset_sample.jsonl"
    with open(sample_path, "w") as f:
        for i in range(min(10, len(all_processed_data))):
            f.write(json.dumps(all_processed_data[i]) + "\n")
    print(f"Saved sample to {sample_path}")
    
    # Push to HF
    print(f"Pushing dataset to {output_repo}...")
    hf_dataset.push_to_hub(output_repo, split="src")
    
    print("Done!")

if __name__ == "__main__":
    main() 