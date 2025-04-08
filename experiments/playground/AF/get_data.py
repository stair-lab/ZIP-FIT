import os
from datasets import load_dataset
import json

# Your Hugging Face token
hf_token = ""

# Load the dataset from Hugging Face with authentication
dataset = load_dataset("AI4M/Math-Proof-Mix", 
                       data_files={"train": ["data/train-00000-of-00004.parquet",
                                             "data/train-00001-of-00004.parquet",
                                             "data/train-00002-of-00004.parquet",
                                             "data/train-00003-of-00004.parquet"]},
                       token=hf_token)  # Use the token parameter for authentication

# Get Hugging Face cache directory
hf_cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/")

# Make sure the directory exists
os.makedirs(hf_cache_dir, exist_ok=True)

# Path to save the JSONL file in the cache directory
output_file = os.path.join(hf_cache_dir, 'output_data.jsonl')

# Function to save the dataset to JSONL
def save_to_jsonl(dataset, output_file):
    with open(output_file, 'w') as f:
        for example in dataset:
            json.dump(example, f)
            f.write('\n')

# Save the dataset to the JSONL file in the Hugging Face cache directory
save_to_jsonl(dataset['train'], output_file)

print(f"Dataset saved as {output_file}")