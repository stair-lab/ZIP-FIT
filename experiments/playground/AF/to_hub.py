from datasets import Dataset
from transformers import AutoTokenizer
import json
from huggingface_hub import create_repo

def load_jsonl(file_path, model_id, hf_token, target_token_count=1000000):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=hf_token, 
        trust_remote_code=True
    )
    
    total_tokens = 0
    data = []
    
    # Read the JSONL file
    with open(file_path, 'r') as f:
        for line in f:
            # Parse the JSON object
            seq = json.loads(line)['text']
            
            # Tokenize the sequence
            seq_tokens = len(tokenizer(seq)['input_ids'])
            
            # Check if adding this sequence exceeds the target token count
            if total_tokens + seq_tokens > target_token_count:
                # Calculate remaining tokens
                remaining_tokens = target_token_count - total_tokens
                
                # Truncate the sequence to fit the remaining tokens
                truncated_tokens = tokenizer(seq)['input_ids'][:remaining_tokens]
                truncated_seq = tokenizer.decode(truncated_tokens)
                
                # Add the truncated sequence and exit
                data.append({'text': truncated_seq})
                break
            else:
                # Add the full sequence
                data.append({'text': seq})
                total_tokens += seq_tokens
    print("total tokens: " ,total_tokens)
    print("total data: " ,len(data))
    # Convert the filtered data to a Hugging Face Dataset
    return Dataset.from_list(data)

hf_token = ""

create_repo("AI4M/zipfit-TOP1M-AF", token=hf_token, repo_type="dataset")  # or False if you want it public
file_path = "./zipfit2.jsonl"
model_id = "google/gemma-2-2b"  # Replace with the correct model ID

# Load the dataset with the first 1M tokens
dataset = load_jsonl(file_path, model_id, hf_token)

# Push to Hugging Face Hub
dataset.push_to_hub("AI4M/zipfit-TOP1M-AF", token=hf_token) 